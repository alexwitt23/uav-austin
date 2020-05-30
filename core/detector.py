""" A detector model which wraps around a feature extraction backbone, fpn, and RetinaNet
head.This allows for easy interchangeability during experimentation and a reliable way to
load saved models. """
import collections
import dataclasses
from typing import List
import yaml

import torch
import torchvision

from core import pull_assets
from third_party.efficientdet import bifpn, efficientnet
from third_party.vovnet import vovnet
from third_party.models import (
    fpn,
    postprocess,
    regression,
    anchors,
    retinanet_head,
)


@dataclasses.dataclass
class BiFPN_Params:
    resolution: int
    channels: int
    num_layers: int
    head_convs: int


_EFFICIENT_DETS = {
    "bifpn-b0": BiFPN_Params(512, 64, 3, 3),
    "bifpn-b1": BiFPN_Params(640, 88, 4, 3),
    "bifpn-b2": BiFPN_Params(768, 112, 5, 3),
    "bifpn-b3": BiFPN_Params(896, 160, 6, 4),
    "bifpn-b4": BiFPN_Params(1024, 224, 7, 4),
    "bifpn-b5": BiFPN_Params(1280, 288, 7, 4),
}


class Detector(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_params: dict = None,
        version: str = None,
        use_cuda: bool = torch.cuda.is_available(),
        half_precision: bool = False,
        confidence: float = 0.05,
        num_detections_per_image: int = 3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_cuda = use_cuda
        self.half_precision = half_precision
        self.num_detections_per_image = num_detections_per_image
        self.confidence = confidence

        if model_params is None and version is None:
            raise ValueError("Must supply either model version or backbone to load")

        # If a version is given, download from bintray
        if version is not None:

            # Download the model. This has the yaml containing the backbone.
            model_dir = pull_assets.download_model(
                model_type="detector", version=version
            )

            # Load the config in the package to determine the backbone
            config = yaml.safe_load((model_dir / "config.yaml").read_text())
            self._load_params(config["model"])
        else:
            self._load_params(model_params)

        self.backbone = self._load_backbone(self.backbone)
        self.fpn = self._load_fpn(self.fpn_type, self.backbone.get_pyramid_channels())

        self.anchors = anchors.AnchorGenerator(
            img_height=self.img_height,
            img_width=self.img_width,
            pyramid_levels=self.fpn_levels,
            aspect_ratios=self.aspect_ratios,
            sizes=self.anchor_sizes,
            anchor_scales=self.anchor_scales,
        )

        # Create the retinanet head.
        self.retinanet_head = retinanet_head.RetinaNetHead(
            num_classes,
            in_channels=self.fpn_channels,
            anchors_per_cell=self.anchors.num_anchors_per_cell,
            num_convolutions=self.num_head_convs,
            use_dw=self.retinanet_head_dw,
        )

        if self.use_cuda:
            self.anchors.all_anchors = self.anchors.all_anchors.cuda()
            self.anchors.anchors_over_all_feature_maps = [
                anchors.cuda() for anchors in self.anchors.anchors_over_all_feature_maps
            ]
            self.cuda()

        self.postprocess = postprocess.PostProcessor(
            num_classes=num_classes,
            anchors_per_level=self.anchors.anchors_over_all_feature_maps,
            regressor=regression.Regressor(),
            max_detections_per_image=num_detections_per_image,
            score_threshold=confidence,
        )

        if version is not None:
            self.load_state_dict(
                torch.load(model_dir / "detector-ap30.pt", map_location="cpu")
            )

        self.eval()

    def _load_params(self, config: dict) -> None:
        """Function to parse the model definition params for later building."""
        self.backbone = config.get("backbone", None)
        assert self.backbone is not None, "Please supply a backbone!"

        fpn_params = config.get("fpn", None)
        head_params = config.get("retinanet_head", None)
        assert fpn_params is not None, "Must supply a fpn section in the config."

        self.fpn_type = fpn_params.get("type", None)
        assert self.fpn_type is not None, "Must supply a fpn type."

        if "bifpn" in self.fpn_type:
            params = _EFFICIENT_DETS[self.fpn_type]
            self.fpn_channels = params.channels
            self.num_bifpn = params.num_layers
            self.num_head_convs = params.head_convs
        else:
            self.fpn_channels = fpn_params.get("num_channels", 128)
            self.use_dw = fpn_params.get("use_dw", True)
            self.num_head_convs = head_params.get("num_levels", 3)

        self.fpn_levels = fpn_params.get("levels", [3, 4, 5, 6, 7])
        self.retinanet_head_dw = head_params.get("use_dw", True)

        anchor_params = config.get("anchors", None)
        assert anchor_params is not None, "Please add an anchor section."
        self.aspect_ratios = anchor_params.get("aspect_ratios", [0.5, 1, 2])
        self.anchor_sizes = anchor_params.get("sizes", [16, 32, 64, 128, 256])
        self.anchor_scales = anchor_params.get("scales", [0.75, 1.0, 1.25])

        self.img_height, self.img_width = config.get("img_size", [512, 512])

    def _load_backbone(self, backbone: str) -> torch.nn.Module:
        """Load the supplied backbone."""
        if "efficient" in backbone:
            model = efficientnet.EfficientNet(
                backbone=backbone, num_classes=self.num_classes
            )
        elif "vovnet" in backbone:
            model = vovnet.VoVNet(backbone)
        else:
            raise ValueError(f"Unsupported backbone {backbone}.")

        return model

    def _load_fpn(self, fpn_name: str, features: List[int]) -> torch.nn.Module:
        if "retinanet" in fpn_name:
            fpn_ = fpn.FPN(in_channels=features[-3:], out_channels=self.fpn_channels)
        elif "bifpn" in fpn_name:
            fpn_ = bifpn.BiFPN(
                in_channels=features,
                out_channels=self.fpn_channels,
                num_bifpns=self.num_bifpn,
                num_levels_in=3,
                bifpn_height=len(features),
            )
        return fpn_

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        levels = self.backbone.forward_pyramids(x)
        # Only keep the levels specified during construction.
        levels = collections.OrderedDict(
            [item for item in levels.items() if item[0] in self.fpn_levels]
        )
        levels = self.fpn(levels)
        classifications, regressions = self.retinanet_head(levels)

        if self.training:
            return classifications, regressions
        else:
            return self.postprocess(classifications, regressions)
