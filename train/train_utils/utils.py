import pathlib

import torch
import pycocotools


def save_model(model: torch.nn.Module, save_path: pathlib.Path) -> None:
    torch.save(model.state_dict(), save_path)


def swa_lr_decay(step: int, cycle_len: int, start_lr: float, end_lr: float) -> float:
    """ Linearly decrease the learning rate over the cycle. """
    return start_lr + ((end_lr - start_lr) / cycle_len) * (step % cycle_len)
