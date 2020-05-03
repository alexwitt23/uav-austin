from typing import List

import torch
import numpy as np


class BiFPN(torch.nn.Module):
    """ Implementation of thee BiFPN originally proposed in 
    https://arxiv.org/pdf/1911.09070.pdf. """

    def __init__(self, in_channels: List[int], out_channels: int, num_bifpns: int):
        """ 
        Args:
            in_channels: A list of the incomming number of filters
            for each pyramid level.
            out_channels: The number of features outputted from the
            latteral convolutions. 
        """
        super().__init__()

        # Construct the lateral convs. These take the outputs from the
        # specified pyramid levels and expand to a common number of
        # channels. NOTE no activations.
        self.lateral_convs = []
        for num_channels in in_channels:
            self.lateral_convs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=num_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                    ),
                ).cuda()
            )
        self.lateral_convs = torch.nn.Sequential(*self.lateral_convs)
        # Construct the BiFPN layers.
        self.bifp_layers = torch.nn.ModuleList([])
        for _ in range(num_bifpns):
            self.bifp_layers.append(
                BiFPNBlock(channels=out_channels, num_levels=len(in_channels))
            )
        self.bifp_layers = torch.nn.Sequential(*self.bifp_layers)

        for block in self.bifp_layers.modules():
            for layer in block.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(layer.weight, mode="fan_out")

    def __call__(self, feature_maps: List[torch.Tensor]):
        """ First apply the lateral convolutions to size all the incoming 
        feature layers to the same size. Then pass through the BiFPN blocks.

        Args:
            feature_maps: Feature maps in sorted order of layer. 
        """
        # Apply the lateral convolutions, then the bifpn layer blocks
        laterals = [
            conv(feature_map)
            for conv, feature_map in zip(self.lateral_convs, feature_maps)
        ]

        return self.bifp_layers(laterals)


class BiFPNBlock(torch.nn.Module):
    def __init__(self, channels: int, num_levels: int):
        """
        Args:
            channels: The number of channels in and out.
            num_levels: The number incoming feature pyramid levels.

        """
        super().__init__()
        self.epsilon = 1e-4
        self.num_levels = num_levels
        self.relu = torch.nn.ReLU()
        # Define the weight tensors which will be used for the
        # normalized fusion.
        self.w1 = torch.nn.Parameter(torch.Tensor(2, num_levels))
        self.w2 = torch.nn.Parameter(torch.Tensor(3, num_levels - 2))
        # Create depthwise separable convolutions that will process
        # the input feature maps. The paper also recommends a RELU
        # activation.
        self.pyramid_convolutions = torch.nn.ModuleList()
        for _ in range(num_levels):
            self.pyramid_convolutions.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3,
                        padding=1,
                        groups=channels,
                        bias=False,
                    ),
                    torch.nn.ReLU(inplace=True),
                )
            )

    def __call__(self, input_maps: List[torch.Tensor]):
        """ NOTE: One might find it useful to observe the orignal paper's
        diagram while reading this code. 

        Args:
            feature_maps: A list of the feature maps from each of the
            pyramid levels. Highest to lowest.

        """
        w1 = self.relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon  # normalize
        w2 = self.relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon  # normalize
        # Make a clone of the input list of feature maps. This will allow
        # for easier accumulation across the BiFPN block.
        input_maps_clone = [tensor.clone() for tensor in input_maps]

        # Form the 'top-to-bottom' pathway. See Section 3.1. Start at the
        # highest level (smallest feature map).
        for idx in reversed(range(len(input_maps))):
            # Interpolate the previous pyramid layer, apply weighted fusion, then
            # convolution over the sum.
            weighted_sum = self.w1[0, idx - 1] * torch.nn.functional.interpolate(
                input_maps[idx], size=input_maps[idx - 1].shape[2:], mode="nearest",
            ) + self.w1[1, idx - 1] * input_maps[idx - 1] / (
                self.w1[0, idx - 1] + self.w1[1, idx - 1] + self.epsilon
            )

            # Apply the BiFPN convolution.
            input_maps_clone[idx - 1] = self.pyramid_convolutions[idx](weighted_sum)

        # Form the bottom up layer. The bottom up layer applies maxpooling to
        # decrease the size going up.
        for idx in range(0, len(input_maps) - 2):
            weighted_sum = (
                self.w2[0, idx] * input_maps[idx + 1]
                + self.w2[1, idx] * input_maps_clone[idx + 1]
                + self.w2[2, idx]
                * torch.nn.functional.max_pool2d(input_maps[idx], kernel_size=2)
                / (self.w2[0, idx] + self.w2[1, idx] + self.w2[2, idx] + self.epsilon)
            )

            # Apply the BiFPN convolution.
            input_maps_clone[idx + 1] = self.pyramid_convolutions[idx](weighted_sum)

        input_maps_clone[self.num_levels - 1] = self.w1[
            0, self.num_levels - 1
        ] * torch.nn.functional.max_pool2d(
            input_maps_clone[self.num_levels - 2], kernel_size=2
        ) + self.w1[
            1, self.num_levels - 1
        ] * input_maps[
            self.num_levels - 1
        ] / (
            self.w1[0, self.num_levels - 1]
            + self.w1[1, self.num_levels - 1]
            + self.epsilon
        )
        input_maps_clone[self.num_levels - 1] = self.pyramid_convolutions[idx](
            input_maps_clone[self.num_levels - 1]
        )
        return input_maps_clone
