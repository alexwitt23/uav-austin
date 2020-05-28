""" This is a collections of layers which expand the incoming feature layers
into box regressions and class probabilities. """

import copy
import collections
from typing import Tuple, List

import torch


class SubNetLayer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        residual: bool = True,
    ) -> None:
        """ Simple Subnet Block that allows for adding a residual as done in
        efficiendet implementation.
        Args:
            channels: The number of input filters.
            kernel_size: The kernel to apply depthwise.
            stride: The depthwise stride.
            padding: Padding to add during depthwise.
            residual: Wether to add residual between levels like in efficiendet.
        """

        super().__init__()
        self.residual = residual
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
            torch.nn.ReLU(inplace=True),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        if self.residual:
            out = out + x
        return out


class RetinaNetHead(torch.nn.Module):
    """ This model head contains two components: classification and box regression.
    See the original RetinaNet paper for more details,
    https://arxiv.org/pdf/1708.02002.pdf. """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        anchors_per_cell: int,
        num_convolutions: int = 4,  # Original paper proposes 4 convs
    ) -> None:
        super().__init__()

        # Create the two subnets
        classification_subnet = torch.nn.ModuleList([])
        for idx in range(num_convolutions):
            classification_subnet += [
                SubNetLayer(channels=in_channels, residual=True if idx > 0 else False)
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

        # The regerssion expands the input into (4 * A) channels. So each x,y in the
        # feature map has (4 * A) channels where 4 represents (dx, dy, dw, dh). The
        # regressions for each component of each anchor box.
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
        self, feature_maps: collections.OrderedDict
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """ Applies the regression and classification subnets to each of the
        incoming feature maps. """

        bbox_regressions = [
            self.regression_subnet(level) for level in feature_maps.values()
        ]
        classifications = [
            self.classification_subnet(level) for level in feature_maps.values()
        ]

        return classifications, bbox_regressions
