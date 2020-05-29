""" Wrapper around the pycocotools library for custom metrics evaluation. """

import pathlib
from typing import List

from pycocotools import coco, cocoeval
import numpy as np


def _summarize(
    eval: dict, iou: float, iou_thresholds: np.ndarray, average_precision: bool = True,
) -> dict:

    if average_precision:
        s = eval["precision"]
        # IoU
        t = np.where(iou == iou_thresholds)[0]
        s = s[t]
        s = s[:, :, :, :, :]
    else:
        s = eval["recall"]
        t = np.where(iou == iou_thresholds)[0]
        s = s[t]
        s = s[:, :, :, :]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    return mean_s


def get_metrics(
    labels_path: pathlib.Path,
    predictions_path: pathlib.Path,
    metrics: List[float] = [30, 50, 75],
) -> dict:
    iou_thresholds = np.array(metrics) / 100

    coco_gt = coco.COCO(labels_path)
    coco_predicted = coco_gt.loadRes(str(predictions_path))
    cocoEval = cocoeval.COCOeval(coco_gt, coco_predicted, "bbox")
    cocoEval.params.iouThrs = iou_thresholds
    cocoEval.params.areaRngLbl = "all"
    cocoEval.params.maxDets = [100]
    cocoEval.evaluate()
    cocoEval.accumulate()

    results = {}
    for eval_type in ["ap", "ar"]:
        for iou in metrics:
            results[f"{eval_type}{iou}"] = _summarize(
                cocoEval.eval,
                iou / 100,
                iou_thresholds,
                average_precision=(eval_type == "ap"),
            )
    return results
