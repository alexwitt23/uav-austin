#!/usr/bin/env python3
""" This script generates training data for the object
detector model. The output will be directories of images
plus COCO metadata jsons. """

from typing import List, Tuple
import multiprocessing
import random
import json
import pathlib
import itertools

from tqdm import tqdm
import PIL
from PIL import (
    Image,
    ImageDraw,
    ImageFilter,
    ImageFont,
    ImageOps,
    ImageFile,
    ImageEnhance,
)

from data_generation import generate_config as config
from core import pull_assets

# Get constants from config
NUM_GEN = int(config.NUM_IMAGES)
MAX_SHAPES = int(config.MAX_PER_SHAPE)
FULL_SIZE = config.FULL_SIZE
TARGET_COLORS = config.TARGET_COLORS
ALPHA_COLORS = config.ALPHA_COLORS
COLORS = config.COLORS
CLASSES = config.OD_CLASSES
ALPHAS = config.ALPHAS

_NUM_COMBINATIONS = 20000


def generate_all_images(gen_type: str, num_gen: int, offset: int = 0) -> None:
    """ Generate all combinations of shape, shape color, alpha, and alpha color. """
    images_dir = config.DATA_DIR / gen_type 
    config.DATA_DIR.mkdir(exist_ok=True, parents=True)
    images_dir.mkdir(exist_ok=True, parents=True)

    r_state = random.getstate()
    random.seed(f"{gen_type}{offset}")

    base_shapes = {shape: get_base_shapes(shape) for shape in config.SHAPE_TYPES}

    # There are multiple paramters this data must map.
    # Shape base, shape color, alpha-numeric, and alpha-numeric color
    a = [
        config.SHAPE_TYPES,
        TARGET_COLORS,
        config.ALPHAS,
        ALPHA_COLORS,
        [angle for angle in range(0, 360, 45)],
    ]
    combinations = list(itertools.product(*a))
    random.shuffle(combinations)
    combinations = combinations[:_NUM_COMBINATIONS]
    num_gen = len(combinations)

    numbers = list(range(offset, num_gen + offset))
    crop_xs = random_list(range(0, config.FULL_SIZE[0] - config.CROP_SIZE[0]), num_gen)
    crop_ys = random_list(range(0, config.FULL_SIZE[1] - config.CROP_SIZE[1]), num_gen)

    backgrounds = random_list(get_backgrounds(), num_gen)
    num_targets = 1
    shape_params = []

    for combination in combinations[:_NUM_COMBINATIONS]:
        combination = list(combination)
        font_files = random_list(config.ALPHA_FONTS, num_targets)

        target_colors = [combination[1]] * num_targets

        # Make sure shape and alpha are different colors
        if combination[1] == combination[3]:
            while combination[3] == combination[1]:
                combination[3] = random.choice(ALPHA_COLORS)

        alpha_colors = [combination[3]] * num_targets

        target_rgbs = [random.choice(COLORS[color]) for color in [combination[1]]]
        alpha_rgbs = [random.choice(COLORS[color]) for color in [combination[3]]]

        sizes = random_list(range(30, 65), num_targets)
        xs = random_list(range(65, config.CROP_SIZE[0] - 65, 20), num_targets)
        ys = random_list(range(65, config.CROP_SIZE[1] - 65, 20), num_targets)

        shape_params.append(
            list(
                zip(
                    [combination[0]],
                    base_shapes[combination[0]],
                    [combination[2]],
                    font_files,
                    sizes,
                    [combination[4]],
                    [combination[1]],
                    target_rgbs,
                    [combination[3]],
                    alpha_rgbs,
                    xs,
                    ys,
                )
            )
        )
    # Put everything into one large iterable so that we can split up
    # data across thread pools.ImageFile.LOAD_TRUNCATED_IMAGES = True
    data = zip(
        numbers, backgrounds, crop_xs, crop_ys, shape_params, [gen_type] * num_gen,
    )

    random.setstate(r_state)

    # Generate in a pool. If specificed, use a given number of threads.
    with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(generate_single_example, data)
        for _ in tqdm(processes, total=num_gen):
            pass


def generate_single_example(data: zip) -> None:
    """Creates a single full image"""
    (number, background, crop_x, crop_y, shape_params, gen_type,) = data
    data_path = config.DATA_DIR / gen_type 

    background = background.copy()
    background = background.crop(
        (crop_x, crop_y, crop_x + config.CROP_SIZE[0], crop_y + config.CROP_SIZE[1])
    )
    shape_imgs, img_name = create_shape(*shape_params[0])
    full_img = add_shapes(background, shape_imgs, shape_params)
    full_img = full_img.resize(config.DETECTOR_SIZE)

    img_fn = data_path / f"ex{number}_{img_name}{config.IMAGE_EXT}"
    full_img.save(img_fn)


def add_shapes(
    background: PIL.Image.Image, shape_img: PIL.Image.Image, shape_params,
) -> Tuple[List[Tuple[int, int, int, int, int]], PIL.Image.Image]:
    """Paste shapes onto background and return bboxes"""
    shape_bboxes: List[Tuple[int, int, int, int, int]] = []

    for i, shape_param in enumerate(shape_params):

        x = shape_param[-2]
        y = shape_param[-1]
        x1, y1, x2, y2 = shape_img.getbbox()
        bg_at_shape = background.crop((x1 + x, y1 + y, x2 + x, y2 + y))
        bg_at_shape.paste(shape_img, (0, 0), shape_img)
        background.paste(bg_at_shape, (x, y))
        background = background.crop((x1 + x, y1 + y, x2 + x, y2 + y))
        background = background.filter(ImageFilter.SMOOTH_MORE)
    return background.convert("RGB")


def get_backgrounds():
    """Get the background assets"""
    # Can be a mix of .png and .jpg
    for backgrounds_folder in config.BACKGROUNDS_DIRS:
        filenames = list(backgrounds_folder.rglob("*.png"))
        filenames += list(backgrounds_folder.rglob("*.jpg"))

    return [Image.open(img).resize(config.FULL_SIZE) for img in filenames]


def get_base_shapes(shape):
    """Get the base shape images for a given shapes"""
    # For now just using the first one to prevent bad alpha placement
    base_path = config.BASE_SHAPES_DIR / shape / f"{shape}-01.png"
    return [Image.open(base_path)]


def random_list(items, count):
    """Get a list of items with length count"""
    return [random.choice(items) for i in range(0, count)]


def create_shape(
    shape,
    base,
    alpha,
    font_file,
    size,
    angle,
    target_color,
    target_rgb,
    alpha_color,
    alpha_rgb,
    x,
    y,
) -> PIL.Image.Image:
    """Create a shape given all the input parameters"""
    target_rgb = augment_color(target_rgb)
    alpha_rgb = augment_color(alpha_rgb)

    image = get_base(base, target_rgb, size)
    image = strip_image(image)
    image = add_alphanumeric(image, shape, alpha, alpha_rgb, font_file)

    w, h = image.size
    ratio = min(size / w, size / h)
    image = image.resize((int(w * ratio), int(h * ratio)), 1)

    image = rotate_shape(image, shape, angle)
    image = strip_image(image)
    img_name = f"{shape}_{target_color}_{alpha}_{alpha_color}_{angle}"
    return image, img_name


def augment_color(color_rgb):
    """Shift the color a bit"""
    r, g, b = color_rgb
    # r = max(min(r + random.randint(-10, 11), 255), 1)
    # g = max(min(g + random.randint(-10, 11), 255), 1)
    # b = max(min(b + random.randint(-10, 11), 255), 1)
    return (r, g, b)


def get_base(base, target_rgb, size):
    """Copy and recolor the base shape"""
    image = base.copy()
    image = image.resize((256, 256), 1)
    image = image.convert("RGBA")

    r, g, b = target_rgb

    for x in range(image.width):
        for y in range(image.height):

            pr, pg, pb, _ = image.getpixel((x, y))

            if pr != 255 or pg != 255 or pb != 255:
                image.putpixel((x, y), (r, g, b, 255))

    return image


def strip_image(image: PIL.Image.Image) -> PIL.Image.Image:
    """Remove white and black edges"""
    for x in range(image.width):
        for y in range(image.height):

            r, g, b, _ = image.getpixel((x, y))

            if r == 255 and g == 255 and b == 255:
                image.putpixel((x, y), (0, 0, 0, 0))

    image = image.crop(image.getbbox())

    return image


def add_alphanumeric(
    image: PIL.Image.Image,
    shape: str,
    alpha,
    alpha_rgb: Tuple[int, int, int],
    font_file,
) -> PIL.Image.Image:
    # Adjust alphanumeric size based on the shape it will be on
    if shape == "star":
        font_multiplier = 0.14
    if shape == "triangle":
        font_multiplier = 0.5
    elif shape == "rectangle":
        font_multiplier = 0.72
    elif shape == "quarter-circle":
        font_multiplier = 0.60
    elif shape == "semicircle":
        font_multiplier = 0.55
    elif shape == "circle":
        font_multiplier = 0.55
    elif shape == "square":
        font_multiplier = 0.60
    elif shape == "trapezoid":
        font_multiplier = 0.60
    else:
        font_multiplier = 0.55

    # Set font size, select font style from fonts file, set font color
    font_size = int(round(font_multiplier * image.height))
    font = ImageFont.truetype(str(font_file), font_size)
    draw = ImageDraw.Draw(image)

    w, h = draw.textsize(alpha, font=font)

    x = (image.width - w) / 2
    y = (image.height - h) / 2

    # Adjust centering of alphanumerics on shapes
    if shape == "pentagon":
        pass
    elif shape == "semicircle":
        pass
    elif shape == "rectangle":
        pass
    elif shape == "trapezoid":
        y -= 20
    elif shape == "star":
        pass
    elif shape == "triangle":
        x -= 24
        y += 12
    elif shape == "quarter-circle":
        y -= 40
        x += 14
    elif shape == "cross":
        y -= 25
    elif shape == "square":
        y -= 10
    elif shape == "circle":
        x -= random.randint(-15, 15)
        y -= random.randint(-15, 15)
    else:
        pass

    draw.text((x, y), alpha, alpha_rgb, font=font)

    return image


def rotate_shape(image, shape, angle):
    return image.rotate(angle, expand=1)


if __name__ == "__main__":
    # Pull the assets if not present locally.
    pull_assets.pull_all()
    generate_all_images("combinations_train", config.NUM_IMAGES, config.NUM_OFFSET)
    generate_all_images(
        "combinations_val", config.NUM_VAL_IMAGES, config.NUM_VAL_OFFSET
    )
