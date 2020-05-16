""" A FPN similar to the one defined in Dectectron2:
https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/backbone/fpn.py
"""

from typing import List

import torch


class DepthwiseSeparable(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1, groups=channels
            ),
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=1, bias=True
            ),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FPN(torch.nn.Module):
    def __init__(
        self, in_channels: List[int], out_channels: int, num_levels: int = 5
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_in = len(in_channels)
        self.num_levels = num_levels

        # Construct the lateral convolutions to adapt the incoming feature maps
        # to the same channel depth.
        self.lateral_convs = torch.nn.ModuleList(
            [DepthwiseSeparable(channels, out_channels) for channels in in_channels]
        )

        # Construct a convolution per level.
        self.convs = torch.nn.ModuleList(
            [DepthwiseSeparable(out_channels, out_channels) for _ in range(num_levels)]
        )

    def __call__(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """ Take the input feature maps and apply the necessary convolutions to build 
        out the num_levels specified. """

        # First, loop over the incoming layers and proceed as follows: from top to bottom,
        # apply lateral convolution, add with the previous layer (if there is one), and
        # then apply a convolution.
        for idx, level in enumerate(reversed(feature_maps)):

            # Apply lateral conv
            feature_maps[-idx - 1] = self.lateral_convs[-idx - 1](level)

            # Add the previous layer upsampled, if there is one.
            if idx > 0:
                feature_maps[-idx - 1] += torch.nn.functional.interpolate(
                    feature_maps[-idx],
                    level.shape[2:],
                    align_corners=True,
                    mode="bilinear",
                )
            feature_maps[-idx - 1] = self.convs[idx](feature_maps[-idx - 1])

        # If more feature maps are needed to be made, take the top most incoming layer
        # and create the remaining levels.
        for idx in range(self.num_levels - self.num_in):

            # Downsample the current most 'low-res' map, then apply convolution.
            # TODO(alex) do we have bandwidth for this to be conv2d?
            new_level = torch.nn.functional._max_pool2d(
                feature_maps[idx + self.num_in - 1], kernel_size=1, stride=2, padding=0
            )
            # Now apply the convolution
            new_level = self.convs[self.num_in + idx](new_level)

            # Apply relu to every new level but the last.
            if idx != (self.num_levels - self.num_in - 1):
                new_level = torch.nn.functional.relu(new_level)

            feature_maps.append(new_level)

        return feature_maps
