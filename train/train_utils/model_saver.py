import pathlib

import torch


def save_model(model: torch.nn.Module, save_path: pathlib.Path) -> None:
    torch.save(model.state_dict(), save_path)
