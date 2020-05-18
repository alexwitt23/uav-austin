# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
from collections import OrderedDict
import torch
from typing import List

_NORM = False

VoVNet19_slim_dw_eSE = {
    "stem": [64, 64, 64],
    "stage_conv_ch": [64, 80, 96, 112],
    "stage_out_ch": [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True,
}

VoVNet19_dw_eSE = {
    "stem": [64, 64, 64],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": True,
}

VoVNet19_slim_eSE = {
    "stem": [64, 64, 128],
    "stage_conv_ch": [64, 80, 96, 112],
    "stage_out_ch": [112, 256, 384, 512],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": False,
}

VoVNet19_eSE = {
    "stem": [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 3,
    "block_per_stage": [1, 1, 1, 1],
    "eSE": True,
    "dw": False,
}

VoVNet39_eSE = {
    "stem": [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 2, 2],
    "eSE": True,
    "dw": False,
}

VoVNet57_eSE = {
    "stem": [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 1, 4, 3],
    "eSE": True,
    "dw": False,
}

VoVNet99_eSE = {
    "stem": [64, 64, 128],
    "stage_conv_ch": [128, 160, 192, 224],
    "stage_out_ch": [256, 512, 768, 1024],
    "layer_per_block": 5,
    "block_per_stage": [1, 3, 9, 3],
    "eSE": True,
    "dw": False,
}

_STAGE_SPECS = {
    "V-19-slim-dw-eSE": VoVNet19_slim_dw_eSE,
    "V-19-dw-eSE": VoVNet19_dw_eSE,
    "V-19-slim-eSE": VoVNet19_slim_eSE,
    "V-19-eSE": VoVNet19_eSE,
    "V-39-eSE": VoVNet39_eSE,
    "V-57-eSE": VoVNet57_eSE,
    "V-99-eSE": VoVNet99_eSE,
}


def dw_conv3x3(
    in_channels: int,
    out_channels: int,
    module_name: int,
    postfix: int,
    stride: int = 1,
    kernel_size: int = 3,
    padding: int = 1,
):
    """ 3x3 depthwise separable, pointwise linear convolution with padding. """
    return [
        (
            "{}_{}/dw_conv3x3".format(module_name, postfix),
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=out_channels,
                bias=False,
            ),
        ),
        (
            "{}_{}/pw_conv1x1".format(module_name, postfix),
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
            ),
        ),
        ("{}_{}/pw_norm".format(module_name, postfix), torch.nn.BatchNorm2d(out_channels)),
        ("{}_{}/pw_relu".format(module_name, postfix), torch.nn.ReLU(inplace=True)),
    ]


def conv3x3(
    in_channels,
    out_channels,
    module_name,
    postfix,
    stride=1,
    groups=1,
    kernel_size=3,
    padding=1,
):
    """3x3 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", torch.nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", torch.nn.ReLU(inplace=True)),
    ]


def conv1x1(
    in_channels,
    out_channels,
    module_name,
    postfix,
    stride=1,
    groups=1,
    kernel_size=1,
    padding=0,
):
    """1x1 convolution with padding"""
    return [
        (
            f"{module_name}_{postfix}/conv",
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
        ),
        (f"{module_name}_{postfix}/norm", torch.nn.BatchNorm2d(out_channels)),
        (f"{module_name}_{postfix}/relu", torch.nn.ReLU(inplace=True)),
    ]


class Hsigmoid(torch.nn.Module):
    """ A modified sigmoid that removes computational complexity by modeling the
    transcendental function as piecewise linear. """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu6(x + 3.0, inplace=True) / 6.0


class eSEModule(torch.nn.Module):
    """ This is adapted from the efficientnet Squeeze Excitation. The idea is not 
    squeezing the number of channels keeps more information. """
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.avg_pool(x)
        out = self.fc(x)
        out = self.hsigmoid(x)
        return out * x


class _OSA_module(torch.nn.Module):
    def __init__(
        self,
        in_ch,
        stage_ch,
        concat_ch,
        layer_per_block,
        module_name,
        SE=False,
        identity=False,
        depthwise=False,
    ):

        super().__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = torch.nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = torch.nn.Sequential(
                OrderedDict(
                    conv1x1(
                        in_channel, stage_ch, "{}_reduction".format(module_name), "0"
                    )
                )
            )
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(
                    torch.nn.Sequential(
                        OrderedDict(dw_conv3x3(stage_ch, stage_ch, module_name, i))
                    )
                )
            else:
                self.layers.append(
                    torch.nn.Sequential(
                        OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))
                    )
                )
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = torch.nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, "concat"))
        )

        self.ese = eSEModule(concat_ch)

    def forward(self, x):

        identity_feat = x

        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(torch.nn.Sequential):
    def __init__(
        self,
        in_ch,
        stage_ch,
        concat_ch,
        block_per_stage,
        layer_per_block,
        stage_num,
        SE=False,
        depthwise=False,
    ):

        super().__init__()

        if not stage_num == 2:
            self.add_module(
                "Pooling", torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            )

        if block_per_stage != 1:
            SE = False
        module_name = f"OSA{stage_num}_1"
        self.add_module(
            module_name,
            _OSA_module(
                in_ch,
                stage_ch,
                concat_ch,
                layer_per_block,
                module_name,
                SE,
                depthwise=depthwise,
            ),
        )
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            module_name = f"OSA{stage_num}_{i + 2}"
            self.add_module(
                module_name,
                _OSA_module(
                    concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    module_name,
                    SE,
                    identity=True,
                    depthwise=depthwise,
                ),
            )


class VoVNet(torch.nn.Module):
    def __init__(self, model_name: str, input_ch: int = 3) -> None:
        """
        Args:
            input_ch(int) : the number of input channel
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "stage2" ...
        """
        super().__init__()
        self.model_name = model_name
        stage_specs = _STAGE_SPECS[model_name]

        stem_ch = stage_specs["stem"]
        config_stage_ch = stage_specs["stage_conv_ch"]
        config_concat_ch = stage_specs["stage_out_ch"]
        block_per_stage = stage_specs["block_per_stage"]
        layer_per_block = stage_specs["layer_per_block"]
        SE = stage_specs["eSE"]
        depthwise = stage_specs["dw"]

        # Construct the stem.
        conv_type = dw_conv3x3 if depthwise else conv3x3
        stem = conv3x3(input_ch, stem_ch[0], "stem", "1", 2)
        stem += conv_type(stem_ch[0], stem_ch[1], "stem", "2", 1)
        stem += conv_type(stem_ch[1], stem_ch[2], "stem", "3", 2)
        self.add_module("stem", torch.nn.Sequential((OrderedDict(stem))))
        current_stride = 4
        self._out_feature_strides = {"stem": current_stride, "stage2": current_stride}
        self._out_feature_channels = {"stem": stem_ch[2]}

        stem_out_ch = [stem_ch[2]]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        # Add the OSA modules
        self.stage_names = []
        for i in range(4):  # num_stages
            name = "%d" % (i + 2)  # stage 2 ... stage 5
            self.stage_names.append(name)
            self.add_module(
                name,
                _OSA_stage(
                    in_ch_list[i],
                    config_stage_ch[i],
                    config_concat_ch[i],
                    block_per_stage[i],
                    layer_per_block,
                    i + 2,
                    SE,
                    depthwise,
                ),
            )

            self._out_feature_channels[name] = config_concat_ch[i]
            if not i == 0:
                self._out_feature_strides[name] = current_stride = int(
                    current_stride * 2
                )

        # initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward_pyramids(self, x: torch.Tensor) -> List[torch.Tensor]:
        """ A forward pass for detection where different pyramid levels
        are extracted. """
        outputs = {}
        x = self.stem(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
            outputs[name] = x
        return outputs

    def get_pyramid_channels(self) -> List[int]:
        """"""
        return _STAGE_SPECS[self.model_name]["stage_out_ch"]
        