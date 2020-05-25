""" A detector model which wraps around a feature extraction backbone, fpn, and RetinaNet
head.This allows for easy interchangeability during experimentation and a reliable way to
load saved models. """
from typing import List
import yaml

import torch
import torchvision

from core import pull_assets
from third_party.models import (
    efficientdet, 
    vovnet, 
    fpn, 
    bifpn,    
    postprocess,
    regression,
    anchors,
    retinanet_head
)

class Detector(torch.nn.Module):
    def __init__(
        self,
        img_width: int,
        img_height: int,
        num_classes: int,
        backbone: str = None,
        fpn_name: str = None,
        version: str = None,
        use_cuda: bool = True,
        half_precision: bool = False,
        num_detections_per_image: int = 3,
        confidence: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_width = img_width
        self.img_height = img_height
        self.use_cuda = use_cuda
        self.half_precision = half_precision
        self.num_detections_per_image = num_detections_per_image
        self.confidence = confidence

        if backbone is None and version is None:
            raise ValueError("Must supply either model version or backbone to load")

        # If a version is given, download from bintray
        if version is not None:
            # Download the model. This has the yaml containing the backbone.
            model_path = pull_assets.download_model(
                model_type="detector", version=version
            )
            # Load the config in the package to determine the backbone
            config = yaml.safe_load((model_path.parent / "config.yaml").read_text())
            backbone = config.get("model", {}).get("backbone", None)
            # Construct the model, then load the state
            self.backbone = self._load_backbone(backbone)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            # If no version supplied, just load the backbone
            self.backbone = self._load_backbone(backbone)
            self.fpn = self._load_fpn(fpn_name, self.backbone.get_pyramid_channels())

        self.anchors = anchors.AnchorGenerator(
            img_height=img_height,
            img_width=img_width,
            pyramid_levels=[3, 4, 5, 6, 7],
            anchor_scales=[1.0, 1.2599, 1.5874]
        )

        # Create the retinanet head.
        self.retinanet_head = retinanet_head.RetinaNetHead(
            num_classes,
            in_channels=64,
            anchors_per_cell=self.anchors.num_anchors_per_cell,
            num_convolutions=4,
            dropout=0.2,
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
            
        self.eval()

    def _load_backbone(self, backbone: str) -> torch.nn.Module:
        """ Load the supplied backbone. """
        if "efficientdet" in backbone:
            model = efficientdet.EfficientDet(
                backbone=backbone,
                num_classes=self.num_classes,
                use_cuda=self.use_cuda,
                num_detections_per_image=self.num_detections_per_image,
                score_threshold=self.confidence,
            )
        elif backbone == "resnet18":
            model = efficientdet.EfficientDet(
                backbone=backbone,
                num_classes=self.num_classes,
                use_cuda=self.use_cuda,
                num_detections_per_image=self.num_detections_per_image,
                score_threshold=self.confidence,
            )
        elif backbone == "resnet34":
            model = efficientdet.EfficientDet(
                backbone=backbone,
                num_classes=self.num_classes,
                use_cuda=self.use_cuda,
                num_detections_per_image=self.num_detections_per_image,
            )
        elif "vovnet" in backbone:
            model = vovnet.VoVNet("V-19-slim-dw-eSE")
        else:
            raise ValueError(f"Unsupported backbone {backbone}.")

        return model

    def _load_fpn(self, fpn_name: str, features: List[int]) -> torch.nn.Module:
        if "retinanet" in fpn_name:
            fpn_ = fpn.FPN(in_channels=features[-3 :], out_channels=64)
        elif "bifpn" in fpn_name:
            fpn_ = BiFPN.BiFPN(
                in_channels=features,
                out_channels=params[2],
                num_bifpns=params[3],
                num_levels_in=3,
                bifpn_height=5,
            )  
        return fpn_

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        levels = list(self.backbone.forward_pyramids(x).values())[-3 :]
        levels = self.fpn(levels)
        classifications, regressions = self.retinanet_head(levels)

        if self.training:
            return classifications, regressions
        else:
            return self.postprocess(classifications, regressions)
