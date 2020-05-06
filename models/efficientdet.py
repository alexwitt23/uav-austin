from typing import List, Tuple

import torch

from models import efficientnet, BiFPN
from third_party import retinanet_head, postprocess, regression, anchors, resnet

_MODEL_SCALES = {
    # (resolution, backbone, bifpn channels, num bifpn layers, head layers)
    "efficientdet-b0": (512, "efficientnet-b0", 64, 3, 3),
    "efficientdet-b1": (640, "efficientnet-b1", 88, 4, 3),
    "efficientdet-b2": (768, "efficientnet-b2", 112, 5, 3),
    "efficientdet-b3": (896, "efficientnet-b3", 160, 6, 4),
    "efficientdet-b4": (1024, "efficientnet-b4", 224, 7, 4),
    "efficientdet-b5": (1280, "efficientnet-b5", 288, 7, 4),
}


class EfficientDet(torch.nn.Module):
    """ Implementatin of EfficientDet originally described in 
    [1] Mingxing Tan, Ruoming Pang, Quoc Le.
    EfficientDet: Scalable and Efficient Object Detection.
    CVPR 2020, https://arxiv.org/abs/1911.09070 """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientdet-b0",
        anchors_per_cell: int = 4,
        levels: List[int] = [3, 4, 5, 6, 7],
        use_cuda: bool = False,
    ) -> None:
        """ 
        Args:
            params: (bifpn channels, num bifpns, num retina net convs)
        """
        super().__init__()
        self.num_pyramids = len(levels)
        # assert backbone in _MODEL_SCALES, backbone

        self.backbone = efficientnet.EfficientNet(
            _MODEL_SCALES[backbone][1], num_classes=num_classes
        )
        # self.backbone = resnet.resnet34(pretrained=True, progress=True)

        # Get the output feature for the pyramids we need
        features = self.backbone.get_pyramid_channels()
        features = features[-self.num_pyramids :]

        params = _MODEL_SCALES["efficientdet-b0"]
        # Create the BiFPN with the supplied parameter options.
        self.fpn = BiFPN.BiFPN(
            in_channels=features,
            out_channels=params[2],
            num_bifpns=params[3],
            num_levels_in=3,
            bifpn_height=5,
        )
        self.anchors = anchors.AnchorGenerator(
            img_height=params[0],
            img_width=params[0],
            pyramid_levels=levels,
            anchor_scales=[1.0, 1.2599, 1.5874],
            use_cuda=use_cuda,
        )
        # Create the resnet head.
        self.retinanet_head = retinanet_head.RetinaNetHead(
            num_classes,
            in_channels=params[2],
            anchors_per_cell=self.anchors.num_anchors_per_cell,
            num_convolutions=params[4],
        )

        if use_cuda:
            self.anchors.all_anchors = self.anchors.all_anchors.cuda()
            self.anchors.anchors_over_all_feature_maps = [
                anchors.cuda() for anchors in self.anchors.anchors_over_all_feature_maps
            ]

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
            return self.postprocess(classifications, regressions)
