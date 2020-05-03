#!/usr/bin/env python3
"""
import sys
import os
import multiprocessing

from pull_assets import pull_all
from create_detection_data import generate_all_images as create_det_data
from create_clf_data import create_clf_images as create_clf_data
import generate_config as config

if __name__ == "__main__":
    # Try to get env vars from tox build
    try:
        NUM_IMAGES = int(os.environ["NUM_IMAGES"])
        NUM_VAL_IMAGES = int(os.environ["NUM_VAL_IMAGES"])

    except Exception:
        config.NUM_IMAGES = 20
        config.NUM_VAL_IMAGES = 10

    print("Pulling assets")
    pull_all()

    print("Creating detection data.")
    create_det_data("detector_train", NUM_IMAGES)
    create_det_data("detector_val", NUM_VAL_IMAGES)

    print("Creating classification data.")
    create_clf_data("clf_train", NUM_IMAGES)
    create_clf_data("clf_val", NUM_VAL_IMAGES)
"""
