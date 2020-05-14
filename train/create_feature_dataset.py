#!/usr/bin/env python3
""" Script used to create a database of the feature extractor's output
for all possible shape combinations. This is needed for inference. """

import argparse
import time
import pathlib
import yaml

import torch
import cv2

from core.protos import target_type_database_pb2
from core import target_typer
from data_generation import generate_config


def create_database(
    model: torch.nn.Module, image_dir: pathlib.Path, save_path: pathlib.Path
) -> None:

    database = target_type_database_pb2.ShapeCombinations()
    for img in image_dir.glob(f"*{generate_config.IMAGE_EXT}"):

        target = database.targets.add()
        image = torch.Tensor(cv2.imread(str(img))).permute(2, 0, 1).unsqueeze(0)
        out = model(image).squeeze(0).tolist()
        for idx in range(len(out)):
            target.feature.append(out[idx])

    save_path.write_bytes(database.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to create dataset of the features extracted from "
        "all possible shapes"
    )
    parser.add_argument(
        "--model_config",
        required=True,
        type=pathlib.Path,
        help="Path to the model config definition.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=pathlib.Path,
        help="Path to the saved model to load.",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=pathlib.Path,
        help="Path to folder containing all possible target combinations.",
    )
    parser.add_argument(
        "--save_path",
        required=True,
        type=pathlib.Path,
        help="Path to save the protobuf dataset to.",
    )
    args = parser.parse_args()

    config_path = args.model_config.expanduser()
    assert config_path.is_file(), f"Can't find {config_path}."

    # Load the model config
    config = yaml.safe_load(config_path.read_text())
    model_cfg = config["model"]
    model = target_typer.TargetTyper(backbone=model_cfg.get("backbone", None))

    model_path = args.model_path.expanduser()
    assert model_path.is_file(), model_path
    model.model.load_state_dict(torch.load(model_path, map_location="cpu"))

    data_dir = args.input_dir.expanduser()
    assert data_dir.is_dir(), data_dir

    save_path = args.save_path.expanduser()
    save_path.parent.mkdir(exist_ok=True, parents=True)
    create_database(model, data_dir, save_path)
