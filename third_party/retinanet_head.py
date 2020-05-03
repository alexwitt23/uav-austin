import copy
from typing import Tuple, List

import torch
import numpy as np


class RetinaNetHead(torch.nn.Module):
    """ This model head contains two components: classification
    and box regression. See the original RetinaNet paper for 
    more details, https://arxiv.org/pdf/1708.02002.pdf. """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        anchors_per_cell: int,
        num_convolutions: int = 4,  # Original paper proposes 4 convs
    ):
        super().__init__()

        # Create the two subnets
        classification_subnet = torch.nn.ModuleList([])
        for _ in range(num_convolutions):
            classification_subnet += [
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=in_channels,
                ),
                torch.nn.ReLU(inplace=True),
            ]
        # NOTE same architecture between box regression and classification
        regression_subnet = copy.deepcopy(classification_subnet)

        # Here is where the two subnets diverge. The classification net
        # expands the input into (anchors_num * num_classes) filters because it
        # predicts 'the probability of object presence at each spatial postion
        # for each of the A anchors'
        classification_subnet.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes * anchors_per_cell,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        # The regerssion expands the input into (4 * A) where 4 represents
        # the position of the regeressed anchor.
        regression_subnet.append(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=anchors_per_cell * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

        self.regression_subnet = torch.nn.Sequential(*regression_subnet)
        self.classification_subnet = torch.nn.Sequential(*classification_subnet)
        bias_value = -np.log((1 - 0.01) / 0.01)
        for block in self.regression_subnet.modules():
            for layer in block.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    torch.nn.init.constant_(self.regression_subnet[-1].bias, bias_value)

        for block in self.classification_subnet.modules():
            for layer in block.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    torch.nn.init.constant_(
                        self.classification_subnet[-1].bias, bias_value
                    )

    def __call__(
        self, feature_maps: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """ Applies the regression and classification subnets to each of the 
        incoming feature maps. """

        classifications, bbox_regressions = [], []
        for map_ in feature_maps:
            bbox_regressions.append(self.regression_subnet(map_))
            classifications.append(self.classification_subnet(map_))

        return classifications, bbox_regressions
