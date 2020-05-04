""" Contains logic for finding and filtering blobs. """

from typing import Tuple, List

from PIL import Image
import numpy as np

from inference import types


def extract_crops(
    image: Image.Image, size: Tuple[int, int], overlap: int
) -> List[types.BBox]:
    """ Crop an image based on the given size and overlap.
    Args:
        image: image to be cropped.
        size: (height, width) of the crops to extract.
        overlap: overlap between two adjacent tiles.
    Return:
        A list of BBox objects.

    >>> img2 = Image.fromarray(np.zeros((8, 8)))
    >>> len(extract_crops(img2, size=(2, 2), overlap=1))
    64
    >>> img1 = Image.fromarray(np.zeros((10, 10)))
    >>> len(extract_crops(img1, size=(5, 5), overlap=0))
    4
    """
    width, height = image.size
    crops: List[types.BBox] = []

    for y1 in range(0, height, size[0] - overlap):
        for x1 in range(0, width, size[1] - overlap):

            if y1 + size[0] > height:
                y1 = height - size[0]

            if x1 + size[1] > width:
                x1 = width - size[1]

            y2 = y1 + size[0]
            x2 = x1 + size[1]

            box = types.BBox(x1, y1, x2, y2)
            box.image = image.crop((x1, y1, x2, y2))
            crops.append(box)

    return crops


def resize_all(
    image_crops: List[types.BBox], new_size: Tuple[int, int]
) -> List[types.BBox]:
    """ Interpolate crops to new size.
    Args:
        image_crops: iBBox objects to be resized.
        new_size: the new size of the images.
    Return:
        A list of the newly resized images.
    
    >>> img = Image.fromarray(np.zeros((10, 10)))
    >>> imgs = extract_crops(img, size=(5, 5), overlap=0)
    >>> imgs = resize_all(imgs, new_size=(20, 20))
    >>> [bbox.image.size for bbox in imgs]
    [(20, 20), (20, 20), (20, 20), (20, 20)]
    """
    return [
        types.BBox(
            crop.x1, crop.y1, crop.x2, crop.y2, image=crop.image.resize(new_size)
        )
        for crop in image_crops
    ]
