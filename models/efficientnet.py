""" Code to implement an effecientnet in PyTorch. 

The architecture is based on scaling four model parameters: 
depth (more layers), width (more filters per layer), resolution 
(larger input images), and dropout (a regularization technique 
to cause sparse feature learning). """

from typing import Tuple, List
import math

import torch
import numpy as np

_MODEL_SCALES = {
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    "efficientnet-b0": (1.0, 1.0, 224, 0.0),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5),
    "efficientnet-b8": (2.2, 3.6, 672, 0.5),
    "efficientnet-l2": (4.3, 5.3, 800, 0.5),
}

# These are the default parameters for the model's mobile inverted residual
# bottleneck layers
_DEFAULT_MB_BLOCKS_ARGS = [
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "id_skip": True,
        "strides": 1,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 2,
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 2,
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 3,
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 3,
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 5,
        "repeats": 4,
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
    {
        "kernel_size": 3,
        "repeats": 1,
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "id_skip": True,
        "strides": 2,
        "se_ratio": 0.25,
    },
]


class Swish(torch.nn.Module):
    """ Swish activation function presented here:
    https://arxiv.org/pdf/1710.05941.pdf. """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


# TODO(alex) this is confusing. Copied right from original implemenation.
# This has to do with keeping the scaling consistent, but I don't understand
# exactly whats going on.
def round_filters(filters: int, scale: float, min_depth: int = 8) -> int:
    """ Determine the number of filters based on the depth multiplier. """

    filters *= scale
    new_filters = max(min_depth, int(filters + min_depth / 2) // min_depth * min_depth)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += min_depth

    return int(new_filters)


def round_repeats(repeats: int, depth_multiplier: int) -> int:
    """ Round off the number of repeats. This determine how many times
    to repeat a block. """
    return int(math.ceil(depth_multiplier * repeats))


class PointwiseConv(torch.nn.Module):
    """ A pointwise convolutional layer. This apply a 1 x 1 x N filter
    to the input to quickly expand the input without many parameters. """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
            Swish(),
        )
        fan_out = int(in_channels * out_channels)
        for layer in self.layers.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.normal_(
                    layer.weight, mean=0.0, std=np.sqrt(2.0 / fan_out)
                )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DepthwiseConv(torch.nn.Module):
    """ A depthwise convolutions with has only one filter per incoming channel
    number. The output channels number is the same and the input. """

    def __init__(self, channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        # Calculate the padding to keep the input 'same' dimensions. Add
        # padding to the right and bottom first.
        diff = kernel_size - stride
        padding = [
            diff // 2,
            diff - diff // 2,
            diff // 2,
            diff - diff // 2,
        ]

        self.layers = torch.nn.Sequential(
            torch.nn.ZeroPad2d(padding),
            torch.nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=channels,
                bias=True,
            ),
            torch.nn.BatchNorm2d(num_features=channels),
            Swish(),
        )
        fan_out = int(kernel_size * channels ** 2)
        for layer in self.layers.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.normal_(
                    layer.weight, mean=0.0, std=np.sqrt(2.0 / fan_out)
                )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SqueezeExcitation(torch.nn.Module):
    """  See here for one of the original implementations:
    https://arxiv.org/pdf/1709.01507.pdf. The layer 'adaptively recalibrates 
    channel-wise feature responses by explicitly  modeling interdependencies
    between channels.' """

    def __init__(
        self, expanded_channels: int, in_channels: int, se_ratio: float
    ) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),  # 1 x 1 x in_channels
            torch.nn.Conv2d(
                in_channels=expanded_channels,
                out_channels=max(1, int(in_channels * se_ratio)),  # Squeeze filters
                kernel_size=1,
                stride=1,
                bias=True,
            ),
            Swish(),
            torch.nn.Conv2d(
                in_channels=max(1, int(in_channels * se_ratio)),  # Expand out
                out_channels=expanded_channels,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
            torch.nn.Sigmoid(),
        )
        fan_out = int(expanded_channels ** 2)
        for layer in self.layers.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.normal_(
                    layer.weight, mean=0.0, std=np.sqrt(2.0 / fan_out)
                )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply the squeezing and excitation, then elementwise multiplacation of 
        the excitation 1 x 1 x out_channels tensor. """
        return x * self.layers(x)


class MBConvBlock(torch.nn.Module):
    """ Mobile inverted residual bottleneck layer. """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        expand_ratio: int,
        stride: int,
        se_ratio: int,
        dropout: float,
        skip: bool,
    ) -> None:
        super().__init__()
        self.skip = skip
        self.in_channels = in_channels
        self.out_channels = out_channels

        # NOTE get expanded channels
        expanded_channels = in_channels * expand_ratio
        self.layers = torch.nn.ModuleList([])

        # add expansion layer if expansion required
        if expand_ratio != 1:
            self.layers += [
                PointwiseConv(in_channels=in_channels, out_channels=expanded_channels),
                torch.nn.BatchNorm2d(num_features=expanded_channels),
                Swish(),
            ]
        self.layers += [
            DepthwiseConv(
                channels=expanded_channels, kernel_size=kernel_size, stride=stride
            ),
            SqueezeExcitation(
                expanded_channels=expanded_channels,
                in_channels=in_channels,
                se_ratio=se_ratio,
            ),
            torch.nn.Conv2d(
                in_channels=expanded_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.Dropout(p=dropout),
        ]
        self.layers = torch.nn.Sequential(*self.layers)

        fan_out = int(expanded_channels * out_channels)
        for layer in self.layers.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.normal_(
                    layer.weight, mean=0.0, std=np.sqrt(2.0 / fan_out)
                )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        if self.skip and self.in_channels == self.out_channels:
            out += x
        return out


class EfficientNet(torch.nn.Module):
    """ Entrypoint for creating an effecientnet. """

    def __init__(self, backbone: str, num_classes: int) -> None:
        """ Instantiant the EffecientNet. 

        Args:
            scale_params: (width_coefficient, depth_coefficient, resolution, dropout_rate)
        """
        super().__init__()
        scale_params = _MODEL_SCALES[backbone]
        # Add the first layer, a simple 3x3 filter conv layer.
        out_channels = round_filters(32, scale=scale_params[0])
        self.features_list = [out_channels]
        self.model_layers = [
            torch.nn.Sequential(
                torch.nn.ZeroPad2d([0, 1, 0, 1]),
                torch.nn.Conv2d(
                    in_channels=3,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    bias=False,
                ),
                torch.nn.BatchNorm2d(out_channels),
                Swish(),
            )
        ]

        fan_out = int(out_channels * 3 ** 2)
        for layer in self.model_layers[0].modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.normal_(
                    layer.weight, mean=0.0, std=np.sqrt(2.0 / fan_out)
                )

        # Now loop over the MBConv layer params
        for mb_params in _DEFAULT_MB_BLOCKS_ARGS:
            out_channels = round_filters(
                filters=mb_params["filters_out"], scale=scale_params[0]
            )
            in_channels = round_filters(
                filters=mb_params["filters_in"], scale=scale_params[0]
            )
            repeats = round_repeats(mb_params["repeats"], scale_params[1])

            # This first block is removed from the repeats section because
            # it works to decrease the size of the input since stride > 1.
            mb_block = torch.nn.ModuleList(
                [
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=mb_params["kernel_size"],
                        expand_ratio=mb_params["expand_ratio"],
                        stride=mb_params["strides"],
                        se_ratio=mb_params["se_ratio"],
                        dropout=scale_params[-1],
                        skip=mb_params["id_skip"],
                    )
                ]
            )
            in_channels = out_channels
            self.features_list.append(out_channels)
            for _ in range(repeats - 1):
                mb_block.append(
                    MBConvBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=mb_params["kernel_size"],
                        expand_ratio=mb_params["expand_ratio"],
                        stride=1,
                        se_ratio=mb_params["se_ratio"],
                        dropout=scale_params[-1],
                        skip=mb_params["id_skip"],
                    )
                )

            self.model_layers.append(torch.nn.Sequential(*mb_block))

        out_channels = round_filters(1280, scale=scale_params[0])
        self.pre_classification = torch.nn.Sequential(
            PointwiseConv(in_channels=in_channels, out_channels=out_channels),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Dropout(p=scale_params[-1]),
        )

        self.model_head = torch.nn.Linear(
            in_features=out_channels, out_features=num_classes
        )

        fan_out = self.model_head.weight.size(0)
        init_range = 1.0 / math.sqrt(fan_out)
        torch.nn.init.uniform_(self.model_head.weight, -init_range, init_range)
        self.feature_levels = self.model_layers
        self.model_layers = torch.nn.Sequential(*self.model_layers)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        features = self.pre_classification(self.model_layers(x))
        features = features.view(features.shape[0], -1)
        return self.model_head(features)
    
    def forward_pyramids(self, x: torch.Tensor) -> List[torch.Tensor]:
        """ Get the outputs at each level. """
        x1 = self.feature_levels[0](x)
        x2 = self.feature_levels[1](x1)
        x3 = self.feature_levels[2](x2)
        x4 = self.feature_levels[3](x3)
        x5 = self.feature_levels[4](x4)
        x6 = self.feature_levels[5](x5)
        x7 = self.feature_levels[6](x6)
        x8 = self.feature_levels[7](x7)

        return [x1, x2, x3, x4, x5, x6, x7, x8]

    def get_pyramid_channels(self) -> List[int]:
        """ Return the number of channels from each pyramid level. We only care 
        about the output channels of each MBConv block. """

        return self.features_list

    def delete_classification_head(self) -> None:
        del self.pre_classification
        del self.model_head
