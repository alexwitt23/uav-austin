import copy
from typing import Tuple, List

import torch
import numpy as np


class SubNetLayer(torch.nn.Module):
    """ Simple Subnet Block that allows for adding a residual as done in
    efficiendet implementation. """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        residual: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.dropout = dropout
        self.layers = torch.nn.Sequential(
            # Depthwise
            torch.nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            # Pointwise linear
            torch.nn.Conv2d(
                in_channels=channels, out_channels=channels, kernel_size=1, bias=True,
            ),
            torch.nn.BatchNorm2d(channels),
            torch.nn.Dropout(p=self.dropout if residual else 0, inplace=True),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        if self.residual:
            out += x
        return out


class RetinaNetHead(torch.nn.Module):
    """ This model head contains two components: classification
    and box regression. See the original RetinaNet paper for 
    more details, https://arxiv.org/pdf/1708.02002.pdf. """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        anchors_per_cell: int,
        num_convolutions: int,  # Original paper proposes 4 convs
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Create the two subnets
        classification_subnet = torch.nn.ModuleList([])
        for idx in range(num_convolutions):
            classification_subnet += [
                SubNetLayer(
                    channels=in_channels,
                    residual=True if idx > 0 else False,
                    dropout=dropout,
                )
            ]
        # NOTE same architecture between box regression and classification
        regression_subnet = copy.deepcopy(classification_subnet)

        # Here is where the two subnets diverge. The classification net
        # expands the input into (anchors_num * num_classes) filters because it
        # predicts 'the probability of object presence at each spatial postion
        # for each of the A anchors'
        classification_subnet += [
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes * anchors_per_cell,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
        ]

        # The regerssion expands the input into (4 * A) where 4 represents
        # the position of the regeressed anchor.
        regression_subnet += [
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=anchors_per_cell * 4,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
        ]

        self.regression_subnet = torch.nn.Sequential(*regression_subnet)
        self.classification_subnet = torch.nn.Sequential(*classification_subnet)

    def __call__(
        self, feature_maps: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """ Applies the regression and classification subnets to each of the 
        incoming feature maps. """

        bbox_regressions = [self.regression_subnet(level) for level in feature_maps]
        classifications = [self.classification_subnet(level) for level in feature_maps]

        return classifications, bbox_regressions
