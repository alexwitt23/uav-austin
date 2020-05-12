#!/usr/bin/env python3
""" Functions to pull in data from the cloud. """
import tarfile
import pathlib
import tempfile
from typing import List, Union

import requests

from data_generation import generate_config as config


def pull_all() -> None:
    """ Pull all assets. """
    pull_backgrounds()
    pull_base_shapes()
    pull_fonts()


def pull_backgrounds() -> None:
    """Pull the shape generation backgrounds."""
    download_file(config.BACKGROUNDS_URL, config.ASSETS_DIR)


def pull_base_shapes() -> None:
    """Pull the base shape images."""
    download_file(config.BASE_SHAPES_URL, config.ASSETS_DIR)


def pull_fonts() -> None:
    """Pull the fonts."""
    download_file(config.FONTS_URL, config.ASSETS_DIR)


# Download a file to the assets folder and return the filename.
def download_file(filenames: Union[str, List[str]], destination: pathlib.Path) -> None:

    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:
        folder_name = filename.split(".", 1)[0]
        if not (destination / folder_name).is_dir():
            url = f"{config._DOWNLOAD_BASE}{filename}"
            print(f"Fetching {filename}...", end="", flush=True)
            res = requests.get(str(url), stream=True)

            with tempfile.TemporaryDirectory() as d:
                tmp_file = pathlib.Path(d) / "file.tar.gz"

                tmp_file.write_bytes(res.raw.read())
                untar_and_move(tmp_file, destination)

            print(" done.")


# Untar a file, unless the directory already exists.
def untar_and_move(filename: pathlib.Path, destination: pathlib.Path) -> None:

    print(f"Extracting {filename.name}...", end="", flush=True)
    with tarfile.open(filename) as tar:
        tar.extractall(destination)

    # Remove hidden files that might have been left behind by
    # the untarring.
    for filename in destination.rglob("._*"):
        filename.unlink()


def download_model(model_type: str, version: str) -> pathlib.Path:
    assert model_type in ["classifier", "detector"], f"Unsupported model {model_type}."
    filename = f"{model_type}-{version}"
    if not (config.ASSETS_DIR / filename).is_dir():
        download_file(f"{filename}.tar.gz", config.ASSETS_DIR / filename)
    return config.ASSETS_DIR / filename / f"{model_type}.pt"
