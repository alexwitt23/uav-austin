""" Datasets for loading classification or detection data. """

from typing import Tuple
import pathlib
import json

import albumentations
from PIL import Image
import torch
import numpy as np


def classification_augmentations(height: int, width: int) -> albumentations.Compose:
    return albumentations.Compose(
        [
            albumentations.Resize(height=height, width=width),
            albumentations.Flip(),
        ]
    )


def detection_augmentations(height: int, width: int) -> albumentations.Compose:
    return albumentations.Compose(
        [
            albumentations.Resize(height=height, width=width),
            albumentations.Normalize(),
        ],
        bbox_params=albumentations.BboxParams(
            format="coco", label_fields=["category_id"]
        ),
    )


class ClfDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: pathlib.Path, img_ext: str = ".png"):
        super().__init__()
        self.images = sorted(list(data_dir.glob(f"*{img_ext}")))
        assert self.images, f"No images found in {data_dir}."

        self.len = len(self.images)
        self.transform = classification_augmentations(224, 244)
        self.data_dir = data_dir

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = np.asarray(Image.open(self.images[idx]).convert("RGB"))
        image = torch.Tensor(self.transform(image=image)["image"])
        image = image.permute(2, 0, 1)
        class_id = 0 if "background" in self.images[idx].stem else 1

        return image, class_id

    def __len__(self) -> int:
        return self.len


class DetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: pathlib.Path,
        metadata_path: pathlib.Path,
        img_ext: str = ".png",
        img_width: int = 512,
        img_height: int = 512,
    ) -> None:
        super().__init__()
        self.meta_data = json.loads(metadata_path.read_text())
        self.images = list(data_dir.glob(f"*{img_ext}"))
        assert self.images, f"No images found in {data_dir}."

        self.len = len(self.images)
        self.transform = detection_augmentations(img_height, img_width)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = np.asarray(Image.open(self.images[idx]).convert("RGB"))
        labels = json.loads(self.images[idx].with_suffix(".json").read_text())

        boxes = [np.array(list(item.values())[1:]) for item in labels["bboxes"]]
        for box in boxes:
            box[2:] += box[:2]

        category_ids = [list(item.values())[0] for item in labels["bboxes"]]

        augmented = self.transform(
            **{"image": image, "bboxes": boxes, "category_id": category_ids}
        )
        boxes = torch.stack([torch.Tensor(dims) for dims in augmented["bboxes"]])
        image = torch.Tensor(augmented["image"]).permute(2, 0, 1)
        # Image coordinates
        boxes = boxes * torch.Tensor(2 * list(image.shape[1:]))

        return image, boxes, torch.Tensor(augmented["category_id"]), idx

    def __len__(self) -> int:
        return self.len
