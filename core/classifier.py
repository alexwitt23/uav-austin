""" A classifier model which wraps around a backbone. This setup allows for easy
interchangeability during experimentation and a reliable way to load saved models. """

import yaml

import torch

from core import pull_assets
from third_party.models import resnet
from third_party.efficientdet import efficientnet
from third_party.vovnet import vovnet


class Classifier(torch.nn.Module):
    def __init__(
        self,
        img_width: int,
        img_height: int,
        num_classes: int = 2,
        version: str = None,
        backbone: str = None,
        use_cuda: bool = False,
        half_precision: bool = False,
    ) -> None:
        """
        Args:
            img_width: The width of the input images.
            img_height: The height of the input images.
            num_classes: The number of classes to predict.
            version: The version of the model to download from bintray.
            backbone: A string designating which model to load.
            use_cuda: Wether this model is going to be used on gpu.
            half_precision: Wether to use half precision for inference. For now
                half_precision doesn't work well with training. Maybe in PyTorch 1.6.0.
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_cuda = use_cuda
        self.half_precision = half_precision
        if backbone is None and version is None:
            raise ValueError("Must supply either model version or backbone to load")

        # If a version is given, download from bintray
        if version is not None:
            # Download the model. This has the yaml containing the backbone.
            model_path = pull_assets.download_model(
                model_type="classifier", version=version
            )
            # Load the config in the package to determine the backbone
            config = yaml.safe_load((model_path / "config.yaml").read_text())
            backbone = config.get("model", {}).get("backbone", None)
            # Construct the model, then load the state
            self.model = self._load_backbone(backbone)
            self.load_state_dict(
                torch.load(model_path / "classifier.pt", map_location="cpu")
            )
        else:
            # If no version supplied, just load the backbone
            self.model = self._load_backbone(backbone)

        self.model.eval()
        if self.use_cuda and self.half_precision:
            self.model.cuda()
            self.model.half()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # If using cuda and not training, assume inference.
        if self.use_cuda and self.half_precision:
            x = x.half()
        return self.model(x)

    def _load_backbone(self, backbone: str) -> torch.nn.Module:
        """ Load the supplied backbone. """
        if backbone in efficientnet._MODEL_SCALES:
            model = efficientnet.EfficientNet(
                backbone=backbone, num_classes=self.num_classes
            )
        elif backbone == "resnet18":
            model = resnet.resnet18(num_classes=self.num_classes)
        elif "vovnet" in backbone:
            model = vovnet.VoVNet(backbone, num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported backbone {backbone}.")

        return model

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """ Take in an image batch and return the class for each image. """
        if self.use_cuda and self.half_precision:
            x = x.half()
        _, predicted = torch.max(self.model(x).data, 1)
        return predicted
