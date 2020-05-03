from typing import List, Tuple

import torch

from models import efficientnet, BiFPN
from third_party import (
    retinanet_head,
    postprocess,
    regression,
    anchors,
)


class EfficientDet(torch.nn.Module):
    """ Implementatin of EfficientDet originally described in 
    [1] Mingxing Tan, Ruoming Pang, Quoc Le.
    EfficientDet: Scalable and Efficient Object Detection.
    CVPR 2020, https://arxiv.org/abs/1911.09070 """

    def __init__(
        self,
        num_classes: int,
        params: Tuple[int, int, int],
        backbone: str = "efficientnet-b0",
        anchors_per_cell: int = 4,
        levels: List[int] = [3, 4, 5, 6, 7],
        img_width: int = 512,
        img_height: int = 512,
    ):
        """ 
        Args:
            params: (bifpn channels, num bifpns, num retina net convs)
        """
        super().__init__()
        self.num_pyramids = len(levels)
        self.backbone = efficientnet.EfficientNet(backbone, num_classes=num_classes)

        # Get the output feature for the pyramids we need
        features = self.backbone.get_pyramid_channels()
        features = features[-self.num_pyramids :]

        # Creat the BiFPN with the supplied parameter options.
        self.fpn = BiFPN.BiFPN(
            in_channels=features, out_channels=params[0], num_bifpns=params[1]
        ).cuda()
        self.anchors = anchors.AnchorGenerator(
            img_height=img_height,
            img_width=img_width,
            pyramid_levels=levels,
            anchor_scales=[1.0, 1.2599, 1.5874],
        )

        # Create the resnet head.
        self.retinanet_head = retinanet_head.RetinaNetHead(
            num_classes,
            in_channels=params[0],
            anchors_per_cell=self.anchors.num_anchors_per_cell,
            num_convolutions=params[2],
        )
        self.postprocess = postprocess.PostProcessor(
            num_classes=num_classes,
            anchors=self.anchors,
            regressor=regression.Regressor(),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        levels = self.backbone.forward_pyramids(x)[-self.num_pyramids :]
        levels = self.fpn(feature_maps=levels)
        classifications, regressions = self.retinanet_head(levels)

        if self.training:
            return classifications, regressions
        else:
            classifications = [level.cpu() for level in classifications]
            regressions = [level.cpu() for level in regressions]
            return self.postprocess(classifications, regressions)
