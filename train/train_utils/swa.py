""" Taken from https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/ but
cleaned up a little bit for readibility. """

import copy

import torch


class SWA(torch.optim.Optimizer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        swa_start: int = 5,
        swa_frequency: int = 5,
    ) -> None:
        """ Implementation of Stochastic Weight Averaging (SWA) based on the paper
        `Averaging Weights Leads to Wider Optima and Better Generalization`_ by Pavel
        Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon 
        Wilson (UAI 2018). 
        
        This optimizer works by ...
        
        Args:
            optimizer: The optimizer that will update the weights of the non-swa model.
            swa_start: Number of optimizer steps after to start updating the SWA weights.
            swa_frequency: After how many optimizer steps to update the weights after 
                :swa_start: is reached.
        """
        self.swa_start = swa_start
        self.swa_frequency = swa_frequency
        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        # Keep a second copy of the model. The SWA copy.
        self.model = model
        self.swa_model = copy.deepcopy(model)
        self.swa_model.cpu()
        self.swa_model.eval()

        # Add a step_counter to keep track of the number of optimization steps. n_avg will
        # keep track of the number effective models the SWA weights contain.
        self.step_counter = 0
        self.n_avg = 0

    def step(self) -> None:
        """ Performs the optimization step for SWA and the original optimizer. """
        # Update the base model's weights.
        self.optimizer.step()
        self.step_counter += 1
        # Update if past starting amount of steps and start of new cycle.
        if (
            self.step_counter > self.swa_start
            and self.step_counter % self.swa_frequency == 0
        ):
            self._update_swa_model()

    def _update_swa_model(self) -> None:
        """ Loop over all the layers from the non-swa model and update their SWA 
        counterparts. """
        with torch.no_grad():
            for layer_id, layer_weights in self.model.model.state_dict().items():
                swa_layer = self.swa_model.model.state_dict()[layer_id]
                swa_layer += (swa_layer * self.n_avg + layer_weights.cpu()) / (
                    self.n_avg + 1
                )

        self.n_avg += 1

    def update_bn(self, loader: torch.utils.data.DataLoader) -> None:
        """ Update the batch norm momumentum for the SWA model since the SWA model's BN params
        are not accoutned for in the weighted average calculation. This function should be called
        before saving the swa model or before swa evaluation. """
        self.swa_model.train()
        if torch.cuda.is_available():
            self.swa_model.cuda()
        momenta = {}
        self.swa_model.apply(reset_bn)
        self.swa_model.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        for data, _ in loader:
            batch_size = data.shape[0]

            momentum = batch_size / (n + batch_size)
            for module in momenta.keys():
                module.momentum = momentum
            if torch.cuda.is_available():
                data = data.cuda()
            self.swa_model(data)
            n += batch_size

        self.swa_model.apply(lambda module: _set_momenta(module, momenta))
        self.swa_model.cpu()
        self.swa_model.eval()


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]
