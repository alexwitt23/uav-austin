from typing import List

import torch
import numpy as np


class BiFPN(torch.nn.Module):
    """ Implementation of thee BiFPN originally proposed in 
    https://arxiv.org/pdf/1911.09070.pdf. """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_bifpns: int,
        num_levels_in: int,
        bifpn_height: int = 5,
    ) -> None:
        """ 
        Args:
            in_channels: A list of the incomming number of filters
                for each pyramid level.
            out_channels: The number of features outputted from the
                latteral convolutions. 
            num_bifpns: The number of BiFPN layers in the model.
                start_level: Which pyramid level to start at.
            num_levels_in: The number of feature maps incoming.
            bifpn_height: The number of feature maps to send in to the
            bifpn. NOTE might not be equal to num_levels_in. 
        """
        super().__init__()
        self.num_levels_in = num_levels_in
        self.bifpn_height = bifpn_height
        self.in_channels = in_channels
        # Construct the lateral convs. These take the outputs from the
        # specified pyramid levels and expand to a common number of
        # channels. NOTE no activations.
        self.lateral_convs = [
            torch.nn.Conv2d(
                in_channels=num_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            )
            for num_channels in in_channels[-num_levels_in:]
        ]
        self.lateral_convs = torch.nn.Sequential(*self.lateral_convs)

        # Construct the BiFPN layers. If we are to take fewer feature pyramids
        # than the list given, we must interpolate the others. This occurs when
        # the supplied feature list might not align with anchor grid generated
        # since the anchor grid assumes that each level is 1 / 2 the W, H of
        # the previous level.
        self.bifp_layers = [
            BiFPNBlock(channels=out_channels, num_levels=bifpn_height)
            for _ in range(num_bifpns)
        ]
        self.bifp_layers = torch.nn.Sequential(*self.bifp_layers)

        # If BiFPN needs more levels than what is being put in, downsample the
        # incoming level. We will expand the number of layers after lateral convs
        if self.bifpn_height != self.num_levels_in:
            self.downsample_convs = [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        groups=out_channels,
                        bias=False,
                    ),
                    torch.nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        bias=True,
                    ),
                )
                for _ in range(self.bifpn_height - num_levels_in)
            ]
            self.downsample_convs = torch.nn.Sequential(*self.downsample_convs)

    def __call__(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """ First apply the lateral convolutions to size all the incoming 
        feature layers to the same size. Then pass through the BiFPN blocks.

        Args:
            feature_maps: Feature maps in sorted order of layer. 
        """
        # Make sure fpn gets the anticipated number of levels.
        assert len(feature_maps) == self.num_levels_in, len(feature_maps)

        # If we need to downsample the last couple layers, first apply a lateral convolution if
        # needed, then upsample. This is similar to the original implementation's
        # `resample_feature_map`:
        # https://github.com/google/automl/blob/3e7d7b77bcefb3f7051de6c468e0e17ce201165e/efficientdet/efficientdet_arch.py#L105.

        # Apply the lateral convolutions, to get the input feature maps to the same
        # number of channels.
        feature_maps = [
            conv(feature_map)
            for conv, feature_map in zip(self.lateral_convs, feature_maps)
        ]
        # Apply the downsampling to form the top layers.
        if self.downsample_convs:
            for layer in self.downsample_convs:
                feature_maps.append(layer(feature_maps[-1]))

        return self.bifp_layers(feature_maps)


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

        # Define the weight tensors which will be used for the normalized fusion.
        self.w1 = torch.nn.Parameter(torch.ones([2, num_levels]) / 10)
        self.w2 = torch.nn.Parameter(torch.ones([3, num_levels - 2]) / 10)

        # Create depthwise separable convolutions that will process
        # the input feature maps. The paper also recommends a RELU activation.
        self.pyramid_convolutions = torch.nn.ModuleList()
        for _ in range(num_levels * 2 - 2):
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
                    torch.nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=1,
                        bias=True,
                    ),
                    torch.nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
                )
            )

    def __call__(self, input_maps: List[torch.Tensor]):
        """ NOTE: One might find it useful to observe the orignal paper's
        diagram while reading this code. 

        Args:
            feature_maps: A list of the feature maps from each of the
            pyramid levels. Highest to lowest.

        """
        conv_idx = 0
        w1 = self.relu(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.epsilon  # normalize
        w2 = self.relu(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.epsilon  # normalize

        # Make a clone of the input list of feature maps. This will allow
        # for easier accumulation across the BiFPN block.
        input_maps_clone = [tensor.clone() for tensor in input_maps]
        # Form the 'top-to-bottom' pathway. See Section 3.1. Start at the
        # highest level (smallest feature map).
        for idx in range(len(input_maps) - 2, 0, -1):
            # Interpolate the previous pyramid layer 'above', apply weighted fusion, then
            # convolution over the sum.
            weighted_sum = w1[0, idx] * torch.nn.functional.interpolate(
                input_maps_clone[idx + 1],
                size=input_maps[idx].shape[2:],
                mode="nearest",
            ) + w1[1, idx] * input_maps[idx] / (torch.sum(w1[0, idx]) + self.epsilon)
            # Apply the BiFPN convolution.
            input_maps_clone[idx] = self.pyramid_convolutions[conv_idx](weighted_sum)
            conv_idx += 1

        # Get the bottom first pyramid layer node
        weighted_sum = w1[0, 0] * torch.nn.functional.interpolate(
            input_maps_clone[1], size=input_maps[0].shape[2:], mode="nearest",
        ) + w1[1, 0] * input_maps[0] / (torch.sum(w1[:, 0]) + self.epsilon)
        input_maps_clone[0] = self.pyramid_convolutions[conv_idx](weighted_sum)
        conv_idx += 1

        # Form the bottom up layer. The bottom up layer applies maxpooling to
        # decrease the size going up.
        for idx in range(1, len(input_maps) - 1):
            if input_maps_clone[idx - 1].shape != input_maps[idx].shape:
                weighted_sum = (
                    w2[0, idx - 1] * input_maps[idx]
                    + w2[1, idx - 1] * input_maps_clone[idx]
                    + w2[2, idx - 1]
                    * torch.nn.functional.max_pool2d(
                        input_maps_clone[idx - 1], kernel_size=2
                    )
                    / (torch.sum(w1[:, idx - 1]) + self.epsilon)
                )
            else:
                weighted_sum = (
                    w2[0, idx - 1] * input_maps[idx]
                    + w2[1, idx - 1] * input_maps_clone[idx]
                    + w2[2, idx - 1]
                    * input_maps_clone[idx - 1]
                    / (torch.sum(w1[:, idx - 1]) + self.epsilon)
                )
            # Apply the BiFPN convolution.
            input_maps_clone[idx] = self.pyramid_convolutions[conv_idx](weighted_sum)
            conv_idx += 1

        weighted_sum = w1[0, self.num_levels - 1] * torch.nn.functional.max_pool2d(
            input_maps_clone[self.num_levels - 2], kernel_size=2
        ) + w1[1, self.num_levels - 1] * input_maps[self.num_levels - 1] / (
            torch.sum(w1[:, self.num_levels - 1]) + self.epsilon
        )
        input_maps_clone[self.num_levels - 1] = self.pyramid_convolutions[conv_idx](
            weighted_sum
        )

        return input_maps_clone
