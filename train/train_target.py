#!/usr/bin/env python3
""" Script used to train a feature extractor model to differentiate
between all combinations of shape, shape color, alpha, and alpha color. """

import argparse
from typing import Tuple
import datetime
import pathlib
import itertools
import random
import yaml
import shutil

import torch
<<<<<<< HEAD
from sklearn import metrics 
=======
from sklearn import metrics
>>>>>>> a20669e7abe4fe90089dd4f4b7aae3a4f4da5690

from core import target_typer
from data_generation import generate_config
from train import datasets
from train_utils import model_saver

_TRIPLET_MARGIN = 10
_LOG_INTERVAL = 1
_SAVE_DIR = pathlib.Path("~/runs/uav-feature-extractor").expanduser()


def train(model_cfg: dict, train_cfg: dict, save_dir: pathlib.Path):

    train_batch_size = train_cfg.get("train_batch_size", 8)
    train_loader = create_data_loader(
        train_cfg, generate_config.DATA_DIR / "combinations_train", train_batch_size,
    )
    eval_batch_size = train_cfg.get("eval_batch_size", 8)
    eval_loader = create_data_loader(
        train_cfg, generate_config.DATA_DIR / "combinations_val", eval_batch_size,
    )

    use_cuda = train_cfg.get("gpu", False)
    save_best = train_cfg.get("save_best", False)
    if save_best:
        highest_score = 0

    model = target_typer.TargetTyper(backbone=model_cfg.get("backbone", None))
    if use_cuda:
        model.cuda()
    model_saver.save_model(model.model, save_dir / "feature_extractor.pt")
    loss_fn = torch.nn.TripletMarginLoss(_TRIPLET_MARGIN)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(20):
        for idx, (anchor, positive, negative) in enumerate(train_loader):

            optim.zero_grad()

            if torch.cuda.is_available():
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            out1 = model(anchor)
            out2 = model(positive)
            out3 = model(negative)
            loss = loss_fn(out1, out2, out3)
            losses.append(loss.item())

            if idx % _LOG_INTERVAL == 0:
                print(f"Epoch: {epoch}. Step {idx} : {sum(losses) / len(losses)}")

            loss.backward()
            optim.step()

        model.eval()
        accuracy = eval(model, eval_loader)
        model.train()

        if accuracy > highest_score:
            model_saver.save_model(model.model, save_dir / "feature_extractor.pt")
            highest_score = accuracy

        print(
            f"Epoch: {epoch}, Training loss {sum(all_losses) / len(all_losses):.5}, "
            f"Eval accuracy: {eval_acc:.4}"
        )


def eval(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> float:
    """ Judge the model's % accuracy based on how many times the anchor and
    positive are within the margin of each other. """
    num_right, total_num = 0, 0
    with torch.no_grad():

        for idx, (anchor, positive, negative) in enumerate(loader):

            if torch.cuda.is_available():
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            out1 = model(anchor)
            out2 = model(positive)
            out3 = model(negative)

            pos_dist = metrics.pairwise.paired_distances(out1.cpu(), out2.cpu())
            neg_dist = metrics.pairwise.paired_distances(out1.cpu(), out3.cpu())

            # Accuracy metric based loss's margin distance
            num_right += (pos_dist + _TRIPLET_MARGIN <= neg_dist).sum().item()
            total_num += pos_dist.shape[0]

        return num_right / total_num


def create_data_loader(
    train_cfg: dict, data_dir: pathlib.Path, batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    assert data_dir.is_dir(), data_dir

    dataset = datasets.TargetDataset(
        pathlib.Path("data_generation/data/combinations_train"),
        img_ext=config.IMAGE_EXT,
        img_width=90,
        img_height=90,
        classes=classes,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=500, pin_memory=True, shuffle=True, num_workers=4
    )
    return loader

    dataset = datasets.TargetDataset(
        pathlib.Path("data_generation/data/combinations_val"),
        img_ext=config.IMAGE_EXT,
        img_width=90,
        img_height=90,
        classes=classes,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset, batch_size=500, pin_memory=True, shuffle=True, num_workers=4
    )
    args = parser.parse_args()

    config_path = args.model_config.expanduser()
    assert config_path.is_file(), f"Can't find {config_path}."

    # Load the model config
    config = yaml.safe_load(config_path.read_text())
    model_cfg = config["model"]
    train_cfg = config["training"]

    save_best = train_cfg.get("save_best", False)
    save_dir = None
    if save_best:
        save_dir = _SAVE_DIR / (datetime.datetime.now().isoformat().split(".")[0])
        save_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(config_path, save_dir / "config.yaml")

    train(model_cfg, train_cfg, save_dir)

    # Create tar archive if best weights are saved.
    if save_best:
        with tarfile.open(save_dir / "detector.tar.gz", mode="w:gz") as tar:
            for model_file in save_dir.glob("*"):
                tar.add(model_file, arcname=model_file.name)
