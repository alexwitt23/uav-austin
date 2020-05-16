#!/usr/bin/env python3
""" Train a classifier to classify images as backgroud or targets. """

import argparse
import datetime
import pathlib
from typing import Tuple
import tarfile
import shutil
import yaml

import torch
from torchcontrib.optim import SWA

from train import datasets
from train.train_utils import model_saver
from core import classifier
from data_generation import generate_config

_LOG_INTERVAL = 50
_SAVE_DIR = pathlib.Path("~/runs/uav-clf").expanduser()


def train(model_cfg: dict, train_cfg: dict, save_dir: pathlib.Path = None) -> None:

    # TODO(alex) these paths should be in the generate config
    train_loader = create_data_loader(train_cfg, generate_config.DATA_DIR / "clf_train")
    eval_loader = create_data_loader(train_cfg, generate_config.DATA_DIR / "clf_val")

    use_cuda = train_cfg.get("gpu", False)
    save_best = train_cfg.get("save_best", False)
    if save_best:
        highest_score = 0.0

    clf_model = classifier.Classifier(
        backbone=model_cfg.get("backbone", None),
        img_width=generate_config.PRECLF_SIZE[0],
        img_height=generate_config.PRECLF_SIZE[0],
        num_classes=2,
    )
    print("Model: \n", clf_model)
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        clf_model.cuda()

    optimizer = create_optimizer(train_cfg["optimizer"], clf_model)
    opt = SWA(optimizer, swa_start=0, swa_lr=1e-5)
    epochs = train_cfg.get("epochs", 0)
    assert epochs > 0, "Please supply epoch > 0"

    # TODO(alex) make this configurable
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_loader), 1e-9
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):

        all_losses = []
        for idx, (data, labels) in enumerate(train_loader):
            opt.zero_grad()

            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()

            out = clf_model(data)
            losses = loss_fn(out, labels)
            all_losses.append(losses.item())
            # Compute the gradient throughout the model graph
            losses.backward()
            # Perform the weight updates
            opt.step()
            # Update the learning rate
            # lr_scheduler.step()

            if idx % _LOG_INTERVAL == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch: {epoch} step {idx}, loss {sum(all_losses) / len(all_losses):.5}. lr: {lr}"
                )

        # Call evaluation function
        opt.swap_swa_sgd()
        clf_model.eval()
        eval_acc = eval(
            clf_model, eval_loader, use_cuda, save_best, highest_score, save_dir
        )
        highest_score = eval_acc if eval_acc > highest_score else highest_score
        clf_model.train()

        print(
            f"Epoch: {epoch}, Training loss {sum(all_losses) / len(all_losses):.5}, "
            f"Eval accuracy: {eval_acc:.4}"
        )
    


def eval(
    clf_model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    use_cuda: bool = False,
    save_best: bool = False,
    previous_best: float = None,
    save_dir: pathlib.Path = None,
) -> float:
    """ Evalulate the model against the evaulation set. Save the best 
    weights if specified. """

    total_num, num_correct = 0, 0
    with torch.no_grad():
        for data, labels in eval_loader:
            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()

            out = clf_model(data)
            _, predicted = torch.max(out.data, 1)

            total_num += labels.size(0)
            num_correct += (predicted == labels).sum().item()

    accuracy = num_correct / total_num

    if save_best and accuracy > previous_best:
        print(f"Saving model with accuracy {accuracy:.5}.")
        # Delete thee previous best
        previous_best = save_dir / "classifier.pt"
        if previous_best.is_file():
            previous_best.unlink()

        model_saver.save_model(clf_model.model, save_dir / "classifier.pt")

    return accuracy


def create_data_loader(
    train_cfg: dict, data_dir: pathlib.Path,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    batch_size = train_cfg.get("batch_size", 64)

    assert data_dir.is_dir(), data_dir

    dataset = datasets.ClfDataset(data_dir, img_ext=generate_config.IMAGE_EXT,)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    return loader


def create_optimizer(optim_cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """ Take in optimizer config and create the optimizer for training. """
    name = optim_cfg.get("type", None)
    if name.lower() == "sgd":
        lr = float(optim_cfg["lr"])
        momentum = float(optim_cfg["momentum"])
        weight_decay = float(optim_cfg["weight_decay"])
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif name.lower() == "rmsprop":
        lr = float(optim_cfg["lr"])
        momentum = float(optim_cfg["momentum"])
        weight_decay = float(optim_cfg["weight_decay"])
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Improper optimizer supplied {name}.")

    return optimizer


if __name__ == "__main__":
    torch.random.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Trainer code for classifcation models."
    )
    parser.add_argument(
        "--model_config",
        required=True,
        type=pathlib.Path,
        help="Path to yaml model definition.",
    )
    args = parser.parse_args()

    config_path = args.model_config.expanduser()
    assert config_path.is_file(), f"Can't find {config_path}."

    # Load the model config
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]

    save_best = train_cfg.get("save_best", False)
    # If save weights, copy in this config file. The config file
    # will be used to load the saved model.
    save_dir = None
    if save_best:
        save_dir = _SAVE_DIR / (datetime.datetime.now().isoformat().split(".")[0])
        save_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(config_path, save_dir / "config.yaml")

    train(model_cfg, train_cfg, save_dir)

    # Create tar archive if best weights are saved.
    if save_best:
        with tarfile.open(save_dir / "classifier.tar.gz", mode="w:gz") as tar:
            for model_file in save_dir.glob("*"):
                tar.add(model_file, arcname=model_file.name)
        print(f"Saved model to {save_dir / 'classifier.tar.gz'}")
