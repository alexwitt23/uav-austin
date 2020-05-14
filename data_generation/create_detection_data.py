#!/usr/bin/env python3
""" This script generates training data for the object
detector model. The output will be directories of images
plus COCO metadata jsons. """

from typing import List, Tuple
import multiprocessing
import random
import json
import pathlib

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


def generate_all_images(gen_type: str, num_gen: int, offset=0) -> None:
    """ Generate the full sized images. """
    images_dir = config.DATA_DIR / gen_type / "images"
    config.DATA_DIR.mkdir(exist_ok=True, parents=True)
    images_dir.mkdir(exist_ok=True, parents=True)

    r_state = random.getstate()
    random.seed(f"{gen_type}{offset}")

    # All the random selection is generated ahead of time, that way
    # the process can be resumed without the shapes changing on each
    # run.

    base_shapes = {shape: get_base_shapes(shape) for shape in config.SHAPE_TYPES}

    numbers = list(range(offset, num_gen + offset))

    backgrounds = random_list(get_backgrounds(), num_gen)
    flip_bg = random_list([False, True], num_gen)
    mirror_bg = random_list([False, True], num_gen)
    sharpen = random_list(range(0, 3), num_gen)
    blurs = random_list(range(1, 2), num_gen)
    num_targets = random_list(range(1, MAX_SHAPES), num_gen)

    crop_xs = random_list(range(0, config.FULL_SIZE[0] - config.CROP_SIZE[0]), num_gen)
    crop_ys = random_list(range(0, config.FULL_SIZE[1] - config.CROP_SIZE[1]), num_gen)

    num_targets = 1
    shape_params = []

    for i in range(num_gen):
        shape_names = random_list(config.SHAPE_TYPES, num_targets)
        bases = [random.choice(base_shapes[shape]) for shape in shape_names]
        alphas = random_list(config.ALPHAS, num_targets)
        font_files = random_list(config.ALPHA_FONTS, num_targets)

        target_colors = random_list(TARGET_COLORS, num_targets)
        alpha_colors = random_list(ALPHA_COLORS, num_targets)

        # Make sure shape and alpha are different colors
        for i, target_color in enumerate(target_colors):
            while alpha_colors[i] == target_color:
                alpha_colors[i] = random.choice(ALPHA_COLORS)

        target_rgbs = [random.choice(COLORS[color]) for color in target_colors]
        alpha_rgbs = [random.choice(COLORS[color]) for color in alpha_colors]

        sizes = random_list(range(20, 65), num_targets)

        angles = random_list(range(0, 360), num_targets)

        xs = random_list(range(70, config.CROP_SIZE[0] - 70), num_targets)
        ys = random_list(range(70, config.CROP_SIZE[1] - 70), num_targets)

        shape_params.append(
            list(
                zip(
                    shape_names,
                    bases,
                    alphas,
                    font_files,
                    sizes,
                    angles,
                    target_colors,
                    target_rgbs,
                    alpha_colors,
                    alpha_rgbs,
                    xs,
                    ys,
                )
            )
        )

    # Put everything into one large iterable so that we can split up
    # data across thread pools.ImageFile.LOAD_TRUNCATED_IMAGES = True
    data = zip(
        numbers,
        backgrounds,
        crop_xs,
        crop_ys,
        flip_bg,
        mirror_bg,
        sharpen,
        blurs,
        shape_params,
        [gen_type] * num_gen,
    )

    random.setstate(r_state)

    # Generate in a pool. If specificed, use a given number of threads.
    with multiprocessing.Pool(None) as pool:
        processes = pool.imap_unordered(generate_single_example, data)
        for _ in tqdm(processes, total=num_gen):
            pass


def generate_single_example(data: zip) -> None:
    """Creates a single full image"""
    (
        number,
        background,
        crop_x,
        crop_y,
        flip_bg,
        mirror_bg,
        sharpen,
        blur,
        shape_params,
        gen_type,
    ) = data

    data_path = config.DATA_DIR / gen_type / "images"
    labels_fn = data_path / f"ex_{number}.json"
    img_fn = data_path / f"ex_{number}{config.IMAGE_EXT}"

    background = background.copy()
    background = background.crop(
        (crop_x, crop_y, crop_x + config.CROP_SIZE[0], crop_y + config.CROP_SIZE[1])
    )

    if flip_bg:
        background = ImageOps.flip(background)
    if mirror_bg:
        background = ImageOps.mirror(background)

    shape_imgs = [create_shape(*shape_param) for shape_param in shape_params]

    shape_bboxes, full_img = add_shapes(background, shape_imgs, shape_params, blur)

    if sharpen == 1:
        full_img = full_img.filter(ImageFilter.SHARPEN)

    full_img = full_img.resize(config.DETECTOR_SIZE)
    full_img.save(img_fn)

    objects = [
        {
            "class_id": shape_bbox[0],
            "x1": shape_bbox[1],
            "y1": shape_bbox[2],
            "w": shape_bbox[3],
            "h": shape_bbox[4],
        }
        for shape_bbox in shape_bboxes
    ]

    labels_fn.write_text(json.dumps({"bboxes": objects, "image_id": number}, indent=2))


def add_shapes(
    background: PIL.Image.Image,
    shape_imgs: PIL.Image.Image,
    shape_params,
    blur_radius: int,
) -> Tuple[List[Tuple[int, int, int, int, int]], PIL.Image.Image]:
    """Paste shapes onto background and return bboxes"""
    shape_bboxes: List[Tuple[int, int, int, int, int]] = []

    for i, shape_param in enumerate(shape_params):

        x = shape_param[-2]
        y = shape_param[-1]
        shape_img = shape_imgs[i]
        shape_img = shape_img.filter(ImageFilter.GaussianBlur(blur_radius))
        x1, y1, x2, y2 = shape_img.getbbox()
        bg_at_shape = background.crop((x1 + x, y1 + y, x2 + x, y2 + y))
        bg_at_shape.paste(shape_img, (0, 0), shape_img)
        background.paste(bg_at_shape, (x, y))

        im_w, im_h = background.size
        x /= im_w
        y /= im_h

        w = (x2 - x1) / im_w
        h = (y2 - y1) / im_h

        shape_bboxes.append((CLASSES.index(shape_param[0]), x, y, w, h))
        shape_bboxes.append(
            (
                CLASSES.index(shape_param[2]),
                x + (0.1 * w),
                y + (0.1 * h),
                0.8 * w,
                0.8 * h,
            )
        )

    return shape_bboxes, background.convert("RGB")


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
    image = ImageEnhance.Color(image).enhance(random.uniform(0.3, 1))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.3, 1))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.3, 1))

    return image


def augment_color(color_rgb):
    """Shift the color a bit"""
    r, g, b = color_rgb
    r = max(min(r + random.randint(-10, 11), 255), 1)
    g = max(min(g + random.randint(-10, 11), 255), 1)
    b = max(min(b + random.randint(-10, 11), 255), 1)
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
        pass
    else:
        pass

    draw.text((x, y), alpha, alpha_rgb, font=font)

    return image


def rotate_shape(image, shape, angle):
    return image.rotate(angle, expand=1)


def create_coco_metadata(data_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    images = []
    annotations = []
    categories = []
    for idx, name in enumerate(CLASSES):
        categories.append({"supercategory": "none", "name": name, "id": idx})

    for idx, label_file in enumerate(data_dir.glob(f"*.json")):
        labels = json.loads(label_file.read_text())
        images.append(
            {
                "file_name": label_file.with_suffix(f"{config.IMAGE_EXT}").name,
                "width": config.DETECTOR_SIZE[0],
                "height": config.DETECTOR_SIZE[1],
                "id": labels["image_id"],
            }
        )
        # Now record the labels
        for label in labels["bboxes"]:
            x1, y1 = (
                int(label["x1"] * config.DETECTOR_SIZE[0]),
                int(label["y1"] * config.DETECTOR_SIZE[1]),
            )
            w, h = (
                int(label["w"] * config.DETECTOR_SIZE[0]),
                int(label["h"] * config.DETECTOR_SIZE[1]),
            )

            annotations.append(
                {
                    "id": len(annotations),
                    "bbox": [
                        int(label["x1"] * config.DETECTOR_SIZE[0]),
                        int(label["y1"] * config.DETECTOR_SIZE[1]),
                        int(label["w"] * config.DETECTOR_SIZE[0]),
                        int(label["h"] * config.DETECTOR_SIZE[1]),
                    ],
                    "category_id": label["class_id"],
                    "iscrowd": 0,
                    "area": int(
                        label["w"]
                        * label["h"]
                        * config.DETECTOR_SIZE[0]
                        * config.DETECTOR_SIZE[0]
                    ),
                    "image_id": labels["image_id"],
                    "segmentation": [[x1, y1, x1, y1 + h, x1 + w, y1 + h, x1 + w, y1]],
                }
            )
    out_path.write_text(
        json.dumps(
            {"categories": categories, "images": images, "annotations": annotations},
            indent=2,
        )
    )


if __name__ == "__main__":
    # Pull the assets if not present locally.
    pull_assets.pull_all()
    generate_all_images("detector_train", config.NUM_IMAGES, config.NUM_OFFSET)
    generate_all_images("detector_val", config.NUM_VAL_IMAGES, config.NUM_VAL_OFFSET)

    create_coco_metadata(
        pathlib.Path(config.DATA_DIR / "detector_train/images"),
        pathlib.Path(config.DATA_DIR / "detector_train/train_coco.json"),
    )

    create_coco_metadata(
        pathlib.Path(config.DATA_DIR / "detector_val/images"),
        pathlib.Path(config.DATA_DIR / "detector_val/val_coco.json"),
    )
