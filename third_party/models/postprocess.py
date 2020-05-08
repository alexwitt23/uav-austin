from typing import List, Tuple
import dataclasses

import torch
from torchvision.ops import boxes as box_ops

from third_party import regression, anchors


@dataclasses.dataclass
class BoundingBox:
    box: torch.Tensor  # [x0, y0, x1, y1]
    confidence: float
    class_id: int  # _internal_ class ID


# https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/retinanet.py
def permute_to_N_HWA_K(retina_tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ Transpose/reshape a tensor from (N, (A x K), H, W) 
    to (N, (HxWxA), K). This is the tensor outputted from the
    RetinaNet head per pyramid level. """

    assert retina_tensor.dim() == 4, retina_tensor.shape
    N, _, H, W = retina_tensor.shape
    retina_tensor = retina_tensor.view(N, -1, num_classes, H, W)
    retina_tensor = retina_tensor.permute(0, 3, 4, 1, 2)
    retina_tensor = retina_tensor.reshape(N, -1, num_classes)  # Size=(N,HWA,K)
    return retina_tensor


def permute_to_N_HWA_K_and_concat(
    box_cls: List[torch.Tensor], box_delta: List[torch.Tensor], num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:

    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_delta
    box_cls_flattened = [permute_to_N_HWA_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HWA_K(x, 4) for x in box_delta]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    return box_cls, box_delta


# https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/nms.py
def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero().view(-1)
        keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


# https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/wrappers.py
def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class PostProcessor:
    def __init__(
        self,
        num_classes: int,
        anchors: anchors.AnchorGenerator,
        regressor: regression.Regressor,
        score_threshold: float = 0.0,
        max_detections_per_image: int = 10,
        nms_threshold: float = 0.5,
        topk_candidates: int = 10,
    ):
        self.num_classes = num_classes
        self.regressor = regressor
        self.anchors = anchors
        self.score_threshold = score_threshold
        self.topk_candidates = topk_candidates
        self.nms_threshold = nms_threshold
        self.max_detections_per_image = max_detections_per_image

    def __call__(
        self,
        box_classifications: List[torch.Tensor],
        box_regressions: List[torch.Tensor],
    ):
        box_classifications = [
            permute_to_N_HWA_K(box_clf, self.num_classes)
            for box_clf in box_classifications
        ]
        box_regressions = [
            permute_to_N_HWA_K(box_reg, 4) for box_reg in box_regressions
        ]
        results = []
        # Loop over the images in the batch
        for img_idx in range(box_classifications[0].shape[0]):
            # Extract out the classification and regression tensors
            box_cls_per_image = [
                box_cls_per_level[img_idx] for box_cls_per_level in box_classifications
            ]
            box_reg_per_image = [
                box_reg_per_level[img_idx] for box_reg_per_level in box_regressions
            ]
            results_per_image = self.inference_single_image(
                box_cls_per_image, box_reg_per_image
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, box_cls, box_delta):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).
        Arguments:
            box_cls (list[Tensor]): list of #feature levels. Each entry contains
                tensor of size (H x W x A, K)
            box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
        Returns:
            Same as `inference`, but for only one image.
        """
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i in zip(
            box_cls, box_delta, self.anchors.anchors_over_all_feature_maps
        ):
            # (HxWxAxK,)
            box_cls_i = box_cls_i.flatten().sigmoid_()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk_candidates, box_reg_i.size(0))
            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
            predicted_prob = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = predicted_prob > self.score_threshold
            predicted_prob = predicted_prob[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = topk_idxs // self.num_classes
            classes_idxs = topk_idxs % self.num_classes

            box_reg_i = box_reg_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]

            # predict boxes
            predicted_boxes = self.regressor.apply_deltas(box_reg_i, anchors_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(predicted_prob)
            class_idxs_all.append(classes_idxs)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_threshold)
        keep = keep[: self.max_detections_per_image]

        return [
            BoundingBox(box.int().cpu(), float(conf), int(cls_id))
            for box, conf, cls_id in zip(
                boxes_all[keep], scores_all[keep], class_idxs_all[keep]
            )
        ]
