#!/usr/bin/env python3
"""Train a classifier to classify images as backgroud or targets."""

import argparse
import datetime
import pathlib
from typing import Tuple
import tarfile
import shutil
import yaml

import torch

from train import datasets
from train.train_utils import utils, swa
from core import classifier
from data_generation import generate_config

_LOG_INTERVAL = 50
_SAVE_DIR = pathlib.Path("~/runs/uav-clf").expanduser()


def train(model_cfg: dict, train_cfg: dict, save_dir: pathlib.Path = None) -> None:

    # TODO(alex) these paths should be in the generate config
    train_loader = create_data_loader(train_cfg, generate_config.DATA_DIR / "clf_train")
    eval_loader = create_data_loader(train_cfg, generate_config.DATA_DIR / "clf_val")

    use_cuda = torch.cuda.is_available()

    highest_score = {"base": 0, "swa": 0}

    clf_model = classifier.Classifier(
        backbone=model_cfg.get("backbone", None),
        img_width=generate_config.PRECLF_SIZE[0],
        img_height=generate_config.PRECLF_SIZE[0],
        num_classes=2,
    )
    print("Model: \n", clf_model)

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        clf_model.cuda()

    optimizer = create_optimizer(train_cfg["optimizer"], clf_model)
    lr_params = train_cfg["lr_scheduler"]

    epochs = train_cfg.get("epochs", 0)
    assert epochs > 0, "Please supply epoch > 0"

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        float(lr_params.get("max_lr", 1e-2)),
        total_steps=len(train_loader) * epochs,
        pct_start=float(lr_params.get("warmup_fraction", 0.1)),
        div_factor=float(lr_params["max_lr"]) / float(lr_params["start_lr"]),
        final_div_factor=float(lr_params["start_lr"]) / float(lr_params["end_lr"]),
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(epochs):
        all_losses = []

        for idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            global_step += 1

            if use_cuda:
                data = data.cuda()
                labels = labels.cuda()

            out = clf_model(data)
            losses = loss_fn(out, labels)
            all_losses.append(losses.item())

            # Compute the gradient throughout the model graph
            losses.backward()
            optimizer.step()
            lr_scheduler.step()

            if idx % _LOG_INTERVAL == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch: {epoch} step {idx}, loss {sum(all_losses) / len(all_losses):.5}. lr: {lr:.4}"
                )

        # Call evaluation function
        clf_model.eval()
        highest_score = eval_acc = eval(
            clf_model, eval_loader, use_cuda, highest_score, save_dir,
        )
        clf_model.train()

        print(
            f"Epoch: {epoch}, Training loss {sum(all_losses) / len(all_losses):.5} \n"
            f"Base accuracy: {eval_acc['base']:.4} \n"
        )


def eval(
    clf_model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    use_cuda: bool = False,
    previous_best: dict = None,
    save_dir: pathlib.Path = None,
) -> float:
    """ Evalulate the model against the evaulation set. Save the best 
    weights if specified. """
    num_correct, total_num = 0, 0

    with torch.no_grad():
        for data, labels in eval_loader:

            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.cuda()

            out = clf_model(data)
            _, predicted = torch.max(out.data, 1)

            num_correct += (predicted == labels).sum().item()
            total_num += data.shape[0]

    accuracy = {
        "base": num_correct / total_num,
    }

    if accuracy["base"] > previous_best["base"]:
        print(f"Saving model with accuracy {accuracy}.")

        # Delete the previous best
        utils.save_model(clf_model, save_dir / "classifier.pt")

        return accuracy
    else:
        return previous_best


def create_data_loader(
    train_cfg: dict, data_dir: pathlib.Path,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    batch_size = train_cfg.get("batch_size", 64)

    assert data_dir.is_dir(), data_dir

    dataset = datasets.ClfDataset(data_dir, img_ext=generate_config.IMAGE_EXT,)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    return loader


def create_optimizer(optim_cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """ Take in optimizer config and create the optimizer for training. """
    name = optim_cfg.get("type", None)
    if name.lower() == "sgd":
        lr = float(optim_cfg["lr"])
        momentum = float(optim_cfg["momentum"])
        weight_decay = float(optim_cfg["weight_decay"])
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif name.lower() == "rmsprop":
        lr = float(optim_cfg["lr"])
        momentum = float(optim_cfg["momentum"])
        weight_decay = float(optim_cfg["weight_decay"])
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Improper optimizer supplied {name}.")

    return optimizer


if __name__ == "__main__":
    torch.random.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.random.manual_seed(42)

    parser = argparse.ArgumentParser(
        description="Trainer code for classifcation models."
    )
    parser.add_argument(
        "--model_config",
        required=True,
        type=pathlib.Path,
        help="Path to yaml model definition.",
    )
    args = parser.parse_args()

    config_path = args.model_config.expanduser()
    assert config_path.is_file(), f"Can't find {config_path}."

    # Load the model config
    config = yaml.safe_load(config_path.read_text())

    model_cfg = config["model"]
    train_cfg = config["training"]

    # Copy in this config file to the save dir. The config file will be used to load the
    # saved model.
    save_dir = _SAVE_DIR / (
        datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")
    )
    save_dir.mkdir(parents=True)
    shutil.copy(config_path, save_dir / "config.yaml")

    train(model_cfg, train_cfg, save_dir)

    # Create tar archive if best weights are saved.
    with tarfile.open(save_dir / "classifier.tar.gz", mode="w:gz") as tar:
        for model_file in save_dir.glob("*"):
            tar.add(model_file, arcname=model_file.name)
    print(f"Saved model to {save_dir / 'classifier.tar.gz'}")
