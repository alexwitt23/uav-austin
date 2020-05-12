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
from data_generation import generate_config

TARGET_COMBINATIONS = {
    "_".join([str(item) for item in name]): idx
    for idx, name in enumerate(
        set(itertools.product(*generate_config.TARGET_COMBINATIONS))
    )
}


def create_batches(
    image_path: np.ndarray,
    tile_size: Tuple[int, int],  # (H, W)
    overlap: int,
    batch_size: int,
) -> Generator[types.BBox]:
    """ Creates batches of images based on the supplied params. """
    image = torch.Tensor(cv2.imread(str(image_path)))
    assert image is not None, f"Could not read {image_path}."

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
            # Figure out the targets
            if boxes:
                target_crops = []


def get_target_type(x: torch.Tensor):
    pass


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
        num_classes=len(generate_config.OD_CLASSES),
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
