#!/usr/bin/env python3
"""Contains logic for finding targets in blobs."""

import argparse
import pathlib
import os
import time
from typing import List, Tuple, Generator
import itertools

import cv2
import numpy as np
import PIL.Image
import torch
from sklearn import metrics

from core import classifier, detector
from inference import preprocessing, types, color_cube
from data_generation import generate_config as config

TARGET_COMBINATIONS = {
    "_".join([str(item) for item in name]): idx
    for idx, name in enumerate(set(itertools.product(*config.TARGET_COMBINATIONS)))
}


def create_batches(
    image: np.ndarray,
    tile_size: Tuple[int, int],  # (H, W)
    overlap: int,
    batch_size: int,
) -> Generator[types.BBox, None, None]:
    """ Creates batches of images based on the supplied params. """
    tiles = []
    coords = []
    for x in range(0, image.shape[1], tile_size[1] - overlap):
        if x + tile_size[1] >= image.shape[1]:
            x = image.shape[0] - tile_size[1]
        for y in range(0, image.shape[0], tile_size[0] - overlap):
            if y + tile_size[0] >= image.shape[0]:
                y = image.shape[0] - tile_size[0]
            tiles.append(
                torch.Tensor(image[y : y + tile_size[0], x : x + tile_size[1]])
            )
            coords.append((x, y))

    tiles = torch.stack(tiles).permute(0, 3, 1, 2)

    for idx in range(0, tiles.shape[0], batch_size):
        yield tiles[idx : idx + batch_size], coords[idx : idx + batch_size]


def find_targets(
    clf_model: torch.nn.Module, det_model: torch.nn.Module, images: List[pathlib.Path],
):
    retval = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        assert image is not None, f"Could not read {image_path}."
        for _ in range(120):
            start = time.perf_counter()
            # Get the image slices.
            for tiles, coords in create_batches(
                image, config.CROP_SIZE, config.CROP_OVERLAP, 100
            ):
                if torch.cuda.is_available():
                    tiles = tiles.cuda().half()

                # Resize the slices for classification.
                tiles = torch.nn.functional.interpolate(tiles, config.PRECLF_SIZE)
                # Call the pre-clf to find the target tiles.
                preds = clf_model.classify(tiles)
                # Get the ids of tiles that contain targets
                target_ids = torch.where(preds == 0)[0].tolist()

                if target_ids:
                    # Pass these target-containing tiles to the detector
                    det_tiles = torch.nn.functional.interpolate(
                        tiles[target_ids], config.DETECTOR_SIZE
                    )
                    boxes = det_model(det_tiles)
                    retval.extend(zip(coords, boxes))
                else:
                    retval.extend(zip(coords, []))

            globalize_boxes(retval)

            print(time.perf_counter() - start)


def globalize_boxes(results):
    final_targets = []
    for coords, bboxes in results:
        if not bboxes:
            continue
        for box in bboxes:
            relative_coords = box.box.tolist()
            relative_coords += list(2 * coords)

            final_targets.append(
                types.Target(
                    x=relative_coords[0],
                    y=relative_coords[1],
                    width=relative_coords[2] - relative_coords[0],
                    height=relative_coords[3] - relative_coords[1],
                )
            )

    return final_targets


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
        img_width=config.PRECLF_SIZE[0],
        img_height=config.PRECLF_SIZE[1],
        use_cuda=torch.cuda.is_available(),
        half_precision=True,
    )
    det_model = detector.Detector(
        version=args.det_version,
        num_classes=len(config.OD_CLASSES),
        img_width=config.DETECTOR_SIZE[0],
        img_height=config.DETECTOR_SIZE[1],
        num_detections_per_image=3,
        use_cuda=torch.cuda.is_available(),
        half_precision=True,
    )

    # Get either the image or images
    if args.image_path is None and args.image_dir is None:
        raise ValueError("Please supply either an image or directory of images.")
    if args.image_path is not None:
        imgs = [args.image_path.expanduser()]
    elif args.image_dir is not None:
        assert args.image_extension.startswith(".")
        imgs = args.image_path.expanduser().glob(f"*{args.image_extension}")

    find_targets(clf_model, det_model, imgs)
