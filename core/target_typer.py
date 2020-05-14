""" A detector model which wraps around a feature extraction
backbone, fpn, and RetinaNet head.This allows for easy 
interchangeability during experimentation and a reliable way 
to load saved models. """

import pathlib
import yaml

import torch
import torchvision

from core import pull_assets
from third_party.models import resnet, efficientnet


class TargetTyper(torch.nn.Module):
    def __init__(
        self,
        version: str = None,
        backbone: str = None,
        use_cuda: bool = False,
        half_precision: bool = False,
    ) -> None:
        super().__init__()
        self.use_cuda = use_cuda
        self.half_precision = half_precision

        if backbone is None and version is None:
            raise ValueError("Must supply either model version or backbone to load")

        # If a version is given, download from bintray
        if version is not None:
            # Download the model. This has the yaml containing the backbone.
            model_path = pull_assets.download_model(
                model_type="target_typer", version=version
            )
            # Load the config in the package to determine the backbone
            config = yaml.safe_load((model_path.parent / "config.yaml").read_text())
            backbone = config.get("model", {}).get("backbone", None)
            # Construct the model, then load the state
            self.model = self._load_backbone(backbone)
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            # If no version supplied, just load the backbone
            self.model = self._load_backbone(backbone)

        self.model.eval()

        if self.use_cuda and self.half_precision:
            self.model.cuda()
            self.model.half()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.final_features(x)

    def _load_backbone(self, backbone: str) -> torch.nn.Module:
        """ Load the supplied backbone. """
        if backbone == "efficientnet-b0":
            model = efficientnet.EfficientNet(backbone=backbone, num_classes=1)
        elif backbone == "resnet18":
            model = resnet.resnet18(num_classes=1)
        else:
            raise ValueError(f"Unsupported backbone {backbone}.")

        model.delete_classification_head()

        return model
