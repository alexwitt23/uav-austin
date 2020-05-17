#!/usr/bin/env python3
"""  Script that will take detection data and copy it for cld-data. """

import multiprocessing

from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

import generate_config as config
from create_detection_data import random_list, get_backgrounds

# Get constants from config
CLF_WIDTH, CLF_HEIGHT = config.PRECLF_SIZE
CROP_WIDTH, CROP_HEIGHT = config.CROP_SIZE


def create_clf_images(gen_type: str, num_gen: int, offset: int = 0) -> None:
    """Generate data for the classifier model."""

    # Make output dir
    save_dir = config.DATA_DIR / gen_type
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get target images
    data_folder = "detector_" + gen_type.split("_")[1]
    images_dir = config.DATA_DIR / data_folder / "images"

    image_names = list(images_dir.glob(f"*{config.IMAGE_EXT}"))
    image_names = random_list(image_names, num_gen)

    numbers = list(range(offset, offset + num_gen))

    # Get random crops and augmentations for background
    backgrounds = random_list(get_backgrounds(), num_gen)
    flip_bg = random_list([False, True], num_gen)
    mirror_bg = random_list([False, True], num_gen)
    blurs = random_list(range(1, 3), num_gen)
    enhancements = random_list(np.linspace(0.5, 2, 5), num_gen)
    crop_xs = random_list(range(0, config.FULL_SIZE[0] - config.CROP_SIZE[0]), num_gen)
    crop_ys = random_list(range(0, config.FULL_SIZE[1] - config.CROP_SIZE[1]), num_gen)

    gen_types = [gen_type] * num_gen

    data = zip(
        numbers,
        backgrounds,
        crop_xs,
        crop_ys,
        flip_bg,
        mirror_bg,
        blurs,
        enhancements,
        image_names,
        gen_types,
    )

    with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(_single_clf_image, data)
        for _ in tqdm(processes, total=num_gen):
            pass


def _single_clf_image(data) -> None:
    """Crop detection image and augment clf image and save"""
    (
        number,
        background,
        crop_x,
        crop_y,
        flip_bg,
        mirror_bg,
        blur,
        enhancement,
        shape_img,
        gen_type,
    ) = data

    background = background.copy()
    background = background.crop(
        (crop_x, crop_y, crop_x + config.CROP_SIZE[0], crop_y + config.CROP_SIZE[1])
    )

    if flip_bg:
        background = ImageOps.flip(background)
    if mirror_bg:
        background = ImageOps.mirror(background)

    background.filter(ImageFilter.GaussianBlur(blur))
    background = background.resize(config.PRECLF_SIZE)
    background = enhance_image(background, enhancement)

    data_path = config.DATA_DIR / gen_type
    bkg_fn = data_path / f"background{number}.{config.IMAGE_EXT}"
    background.save(bkg_fn)

    # Now consider the shape image
    shape = Image.open(shape_img).resize(config.PRECLF_SIZE)
    shape = enhance_image(shape, enhancement)
    shape_fn = data_path / f"target{number}.{config.IMAGE_EXT}"
    shape.save(shape_fn)


def enhance_image(img, enhancement):
    converter = ImageEnhance.Color(img)
    return converter.enhance(enhancement)


if __name__ == "__main__":

    if config.NUM_IMAGES != 0:
        create_clf_images("clf_train", config.NUM_IMAGES, config.NUM_OFFSET)

    if config.NUM_VAL_IMAGES != 0:
        create_clf_images("clf_val", config.NUM_VAL_IMAGES, config.NUM_VAL_OFFSET)
