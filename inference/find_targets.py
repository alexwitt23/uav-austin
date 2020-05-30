#!/usr/bin/env python3
"""Contains logic for finding targets in images."""

import argparse
import pathlib
import time
import json
from typing import List, Tuple, Generator

import cv2
import numpy as np
import torch
from sklearn import metrics

from core import classifier, detector
from inference import types
from data_generation import generate_config as config
from third_party.models import postprocess


def tile_image(
    image: np.ndarray, tile_size: Tuple[int, int], overlap: int  # (H, W)
) -> torch.Tensor:

    tiles = []
    coords = []
    for x in range(0, image.shape[1], tile_size[1] - overlap):

        # Shift back to extract tiles on the image
        if x + tile_size[1] >= image.shape[1]:
            x = image.shape[0] - tile_size[1]

        for y in range(0, image.shape[0], tile_size[0] - overlap):
            if y + tile_size[0] >= image.shape[0]:
                y = image.shape[0] - tile_size[0]

            tiles.append(
                torch.Tensor(image[y : y + tile_size[0], x : x + tile_size[1]])
            )
            coords.append((x, y))

        # Transpose the images from BHWC -> BCHW
        tiles = torch.stack(tiles).permute(0, 3, 1, 2)

    return tiles, coords


def create_batches(
    image_tensor: torch.Tensor, coords: List[Tuple[int, int]], batch_size: int
) -> Generator[types.BBox, None, None]:
    """ Creates batches of images based on the supplied params. The whole image
    is tiled first, the batches are generated.
    Args:
        image: The opencv opened image.
        tile_size: The height, width of the tiles to create.
        overlap: The amount of overlap between tiles.
        batch_size: The number of images to have per batch.
    Returns:
        Yields the image batch and the top left coordinate of the tile in the
        space of the original image.
    """

    for idx in range(0, image_tensor.shape[0], batch_size):
        yield image_tensor[idx : idx + batch_size], coords[idx : idx + batch_size]


def find_targets(
    clf_model: torch.nn.Module,
    det_model: torch.nn.Module,
    images: List[pathlib.Path],
    save_jsons: bool = False,
    visualization_dir: pathlib.Path = None,
) -> None:
    retval = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        assert image is not None, f"Could not read {image_path}."
        start = time.perf_counter()
        image_tensor, coords = tile_image(image, config.CROP_SIZE, config.CROP_OVERLAP)

        # Get the image slices.
        for tiles, coords in create_batches(image_tensor, coords, 30):

            if torch.cuda.is_available():
                tiles = tiles.cuda()

            # Resize the slices for classification.
            tiles = torch.nn.functional.interpolate(tiles, config.PRECLF_SIZE)

            # Call the pre-clf to find the target tiles.
            preds = clf_model.classify(tiles)

            # Get the ids of tiles that contain targets
            target_ids = preds == torch.ones_like(preds)

            if target_ids.sum().item():
                for det_tiles, det_coords in create_batches(
                    tiles[target_ids], coords, 15
                ):
                    # Pass these target-containing tiles to the detector
                    det_tiles = torch.nn.functional.interpolate(
                        det_tiles, config.DETECTOR_SIZE
                    )
                    boxes = det_model(det_tiles)
                    retval.extend(zip(det_coords, boxes))
            else:
                retval.extend(zip(coords, []))

        targets = globalize_boxes(retval)

        print(time.perf_counter() - start)

        if visualization_dir is not None:
            visualize_image(image_path.name, image, visualization_dir, targets)


def globalize_boxes(results: List[postprocess.BoundingBox]) -> List[types.Target]:
    final_targets = []
    for coords, bboxes in results:
        for box in bboxes:
            relative_coords = box.box
            relative_coords += torch.Tensor(list(2 * coords)).int()
            relative_coords = relative_coords.tolist()
            final_targets.append(
                types.Target(
                    x=relative_coords[0],
                    y=relative_coords[1],
                    width=relative_coords[2] - relative_coords[0],
                    height=relative_coords[3] - relative_coords[1],
                )
            )

    return final_targets


def visualize_image(
    image_name: str,
    image: np.ndarray,
    visualization_dir: pathlib.Path,
    targets: List[types.Target],
) -> None:
    """ Function used to draw boxes and information onto image for
    visualizing the output of inference. """
    for target in targets:
        top_left = (target.x, target.y)
        bottom_right = (target.x + target.width, target.y + target.height)
        image = cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 3)

    cv2.imwrite(str(visualization_dir / image_name), image)


def save_target_meta(filename_meta, filename_image, target):
    """ Save target metadata to a file. """
    with open(filename_meta, "w") as f:
        meta = {
            "x": target.x,
            "y": target.y,
            "width": target.width,
            "height": target.height,
            "orientation": target.orientation,
            "shape": target.shape.name.lower(),
            "background_color": target.background_color.name.lower(),
            "alphanumeric": target.alphanumeric,
            "alphanumeric_color": target.alphanumeric_color.name.lower(),
            "image": filename_image,
            "confidence": target.confidence,
        }

        json.dump(meta, f, indent=2)


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
    parser.add_argument(
        "--visualization_dir",
        required=False,
        type=pathlib.Path,
        help="Optional directory to save visualization to.",
    )
    args = parser.parse_args()

    clf_model = classifier.Classifier(
        version=args.clf_version,
        img_width=config.PRECLF_SIZE[0],
        img_height=config.PRECLF_SIZE[1],
        use_cuda=torch.cuda.is_available(),
        half_precision=torch.cuda.is_available(),
    )
    clf_model.eval()
    det_model = detector.Detector(
        version=args.det_version,
        num_classes=len(config.OD_CLASSES),
        confidence=0.99,
        use_cuda=torch.cuda.is_available(),
        half_precision=torch.cuda.is_available(),
    )
    det_model.eval()

    # Get either the image or images
    if args.image_path is None and args.image_dir is None:
        raise ValueError("Please supply either an image or directory of images.")
    if args.image_path is not None:
        imgs = [args.image_path.expanduser()]
    elif args.image_dir is not None:
        assert args.image_extension.startswith(".")
        imgs = args.image_dir.expanduser().glob(f"*{args.image_extension}")

    viz_dir = None
    if args.visualization_dir is not None:
        viz_dir = args.visualization_dir.expanduser()
        viz_dir.mkdir(exist_ok=True, parents=True)

    find_targets(clf_model, det_model, imgs, visualization_dir=viz_dir)
