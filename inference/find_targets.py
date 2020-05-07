#!/usr/bin/env python3
"""Contains logic for finding targets in blobs."""

import argparse
import pathlib
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
import PIL.Image
import torch

from models import classifier, detector
from inference import preprocessing, types, color_cube
from data_generation import generate_config

"""

_IMG_SIZES = generate_config.get("inputs", None)
crop_size = (
    _IMG_SIZES["cropping"]["width"],
    _IMG_SIZES["cropping"]["height"],
)

overlap = _IMG_SIZES["cropping"]["overlap"]

pre_clf_size = (
    _IMG_SIZES["preclf"]["width"],
    _IMG_SIZES["preclf"]["height"],
)

det_size = (
    _IMG_SIZES["detector"]["width"],
    _IMG_SIZES["detector"]["height"],
)
"""


def create_batches(
    image: np.ndarray,
    tile_size: Tuple[int, int],  # (H, W)
    overlap: int,
    batch_size: int,
) -> List[types.BBox]:
    """ Creates batches of images based on the supplied params. """
    image = torch.Tensor(cv2.imread(str(image)))
    image_tiles = (
        image.unfold(0, tile_size[1], tile_size[1] - overlap,)
        .unfold(1, tile_size[0], tile_size[0] - overlap,)
        .unfold(2, image.shape[-1], image.shape[-1])
        .reshape(-1, tile_size[1], tile_size[0], image.shape[-1])
    )
    for idx in range(0, image_tiles.shape[0], batch_size):
        yield image_tiles[idx : idx + batch_size].permute(0, 3, 1, 2)


def find_targets(
    clf_model: torch.nn.Module, det_model: torch.nn.Module, images: List[pathlib.Path],
):
    retval = []
    for image in images:
        start = time.perf_counter()
        for tiles in create_batches(image, (512, 512), 20, 20):
            # Call the pre-clf to find the target tiles.
            preds = clf_model.classify(
                torch.nn.functional.interpolate(tiles, (224, 224))
            )
            # Get the ids of tiles that contain targets
            target_ids = torch.where(preds == 0)[0]
            if target_ids.tolist():
                # Pass these target-positive tiles to the detector
                boxes = det_model(tiles[target_ids])
        print(time.perf_counter() - start)


def _run_models(image):

    crops = extract_crops(image, crop_size, overlap)
    clf_crops = resize_all(crops, pre_clf_size)
    regions = clf_model.predict([box.image for box in clf_crops])

    filtered_crops = []
    for i, region in enumerate(regions):
        if region.class_idx == 1:
            filtered_crops.append(crops[i])

    detector_crops = resize_all(filtered_crops, det_size)

    if len(detector_crops) != 0:
        offset_dets = detector_model.predict([box.image for box in detector_crops])
    else:
        offset_dets = []

    ratio = det_size[0] / crop_size[0]
    normalized_bboxes = []

    for crop, offset_dets in zip(detector_crops, offset_dets):

        for det in offset_dets:

            bw = det.width / ratio
            bh = det.height / ratio
            bx = (det.x / ratio) + crop.x1
            by = (det.y / ratio) + crop.y1
            box = BBox(bx, by, bx + bw, by + bh)
            box.meta = {det.class_name: det.confidence}
            box.confidence = det.confidence
            normalized_bboxes.append(box)

    return normalized_bboxes


def _bboxes_to_targets(bboxes):
    targets = []
    merged_bboxes = _merge_boxes(bboxes)

    for box in merged_bboxes:
        shape, alpha, conf = _get_shape_and_alpha(box)
        targets.append(
            Target(
                box.x1,
                box.y1,
                box.w,
                box.h,
                shape=shape,
                alphanumeric=alpha,
                confidence=conf,
            )
        )

    return targets


"""
def _get_shape_and_alpha(box):

    best_shape, conf_shape = "unk", 0
    best_alpha, conf_alpha = "unk", 0

    for class_name, conf in box.meta.items():
        if len(class_name) == 1 and conf > conf_alpha:
            best_alpha = class_name
            conf_alpha = conf
        elif len(class_name) != 1 and conf > conf_shape:
            best_shape = class_name
            conf_shape = conf

    # convert name to object
    if best_shape == "unk":
        shape = Shape.NAS
    else:
        shape = Shape[best_shape.upper().replace("-", "_")]

    return shape, best_alpha, ((conf_shape + conf_alpha) / 2)


def _merge_boxes(boxes):
    merged = []
    for box in boxes:
        for merged_box in merged:
            if _intersect(box, merged_box):
                _enlarge(merged_box, box)
                merged_box.meta.update(box.meta)
                break
        else:
            merged.append(box)
    return merged


def _intersect(box1, box2):
    # no intersection along x-axis
    if box1.x1 > box2.x2 or box2.x1 > box1.x2:
        return False

    # no intersection along y-axis
    if box1.y1 > box2.y2 or box2.y1 > box1.y2:
        return False

    return True


def _enlarge(main_box, new_box):
    main_box.x1 = min(main_box.x1, new_box.x1)
    main_box.x2 = max(main_box.x2, new_box.x2)
    main_box.y1 = min(main_box.y1, new_box.y1)
    main_box.y2 = max(main_box.y2, new_box.y2)


def _identify_properties(targets, full_image, padding=15):

    for target in targets:

        x = int(target.x) - padding
        y = int(target.y) - padding
        w = int(target.width) + padding * 2
        h = int(target.height) + padding * 2
        blob_image = full_image.crop((x, y, x + w, y + h))

        target.image = full_image

        try:
            target_color, alpha_color = _get_colors(blob_image)
            target.background_color = target_color
            target.alphanumeric_color = alpha_color
        except Exception as e:
            target.background_color = Color.NONE
            target.alphanumeric_color = Color.NONE


def _get_colors(image):

    (color_a, count_a), (color_b, count_b) = _find_main_colors(image)

    # this assumes the shape will have more pixels than alphanum
    if count_a > count_b:
        primary, secondary = color_a, color_b
    else:
        primary, secondary = color_b, color_a

    primary_color = _get_color_name(primary)
    secondary_color = _get_color_name(secondary)

    return primary_color, secondary_color


def _find_main_colors(image):
    # TODO see: https://github.com/uavaustin/target-finder/issues/16
    ar = np.asarray(image)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, 3)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences
    top2 = heapq.nlargest(3, counts)  # find most frequent

    color_a = codes[np.where(counts == top2[0])][0]
    color_a = (color_a[0], color_a[1], color_a[2])
    count_a = top2[0]

    color_b = codes[np.where(counts == top2[1])][0]
    color_b = (color_b[0], color_b[1], color_b[2])
    count_b = top2[1]

    return (color_a, count_a), (color_b, count_b)


def _get_color_name(requested_color):

    # ColorCube((Hl, sl, vl), (Hu, Su, Vu))
    color_cubes = {
        "white": ColorCube((0, 0, 85), (359, 20, 100)),
        "black": ColorCube((0, 0, 0), (359, 100, 25)),
        "gray": ColorCube((0, 0, 25), (359, 5, 75)),
        "blue": ColorCube((180, 70, 70), (345, 100, 100)),
        "red": ColorCube((350, 70, 70), (359, 100, 65)),
        "green": ColorCube((100, 60, 30), (160, 100, 100)),
        "yellow": ColorCube((60, 50, 55), (75, 100, 100)),
        "purple": ColorCube((230, 40, 55), (280, 100, 100)),
        "brown": ColorCube((300, 38, 20), (359, 100, 40)),
        "orange": ColorCube((15, 70, 75), (45, 100, 100)),
    }

    r = requested_color[0] / 255
    g = requested_color[1] / 255
    b = requested_color[2] / 255

    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min

    h = 0
    s = 0
    v = 0
    if delta == 0:
        h = 0
    elif c_max == r:
        h = 60 * (((g - b) / delta) % 6)
    elif c_max == g:
        h = 60 * (((b - r) / delta) + 2)
    elif c_max == b:
        h = 60 * (((r - g) / delta) + 4)

    if c_max == 0:
        s = 0
    else:
        s = delta / c_max
    v = c_max * 100
    s *= 100

    contains = [color_cubes[key].contains((h, s, v)) for key in color_cubes]
    if not any(contains):
        cl_dists = [
            color_cubes[key].get_closest_distance((h, s, v)) for key in color_cubes
        ]
        dist = [
            np.sqrt((p[0] * p[0]) + (p[1] * p[1]) + (p[2] * p[2])) for p in cl_dists
        ]
        index = dist.index(min(dist)) + 1
        return Color(index)
    else:
        index = contains.index(True) + 1
        return Color(index)

    return Color.NONE
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script used to find targets in an image"
    )
    parser.add_argument(
        "--image_path",
        required=False,
        type=pathlib.Path,
        help="Path to an image to inference.",
    )
    parser.add_argument(
        "--image_dir",
        required=False,
        type=pathlib.Path,
        help="Path to directory of images to inference.",
    )
    parser.add_argument(
        "--image_extension",
        required=False,
        type=str,
        help="Needed when an image directory is supplied.",
    )
    parser.add_argument(
        "--visualization_dir",
        required=False,
        type=pathlib.Path,
        help="Directory to save visualizations to.",
    )
    parser.add_argument(
        "--clf_version",
        required=False,
        type=str,
        default="dev",
        help="Version of the classifier model to use.",
    )
    parser.add_argument(
        "--det_version",
        required=False,
        type=str,
        default="dev",
        help="Version of the detector model to use.",
    )
    args = parser.parse_args()

    clf_model = classifier.Classifier(
        version=args.clf_version,
        img_width=generate_config.PRECLF_SIZE[0],
        img_height=generate_config.PRECLF_SIZE[1],
    )
    det_model = detector.Detector(
        version=args.det_version,
        num_classes=37,
        img_width=generate_config.DETECTOR_SIZE[0],
        img_height=generate_config.DETECTOR_SIZE[1],
    )

    # Get either the image or images
    if args.image_path is not None:
        imgs = [args.image_path.expanduser()]
    elif args.image_dir is not None:
        assert args.image_extension.startswith(".")
        imgs = args.image_path.expanduser().glob(f"*{args.image_extension}")

    find_targets(clf_model, det_model, imgs)
