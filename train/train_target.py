#!/usr/bin/env python3
""" Script used to train a feature extractor model to differentiate
between all combinations of shape, shape color, alpha, and alpha color. """

import itertools
import pathlib
import random 

import torch

from core import target_typer
from data_generation import generate_config as config
from train import datasets

_TRIPLET_MARGIN = 10
_LOG_INTERVAL = 1
_NUM_COMBINATIONS = 4000

def train(
    model: torch.nn.Module, 
    train_loader: torch.utils.data.DataLoader,
    eval_loader: torch.utils.data.DataLoader
):
    losses = []
    loss_fn = torch.nn.TripletMarginLoss(_TRIPLET_MARGIN)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
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

        accuracy = eval(model, eval_loader)
        print(f"Eval Accuracy: {accuracy}.")

    torch.save(model.state_dict(), "/home/alex/Desktop/model.pt")


def eval(model: torch.nn.Module, loader: torch.utils.data.DataLoader):
    """ Judge the model's % accuracy based on how many times the anchor and
    positive are within the margin of each other. """

    with torch.no_grad():
        for idx, (anchor, positive, negative) in enumerate(loader):

            if torch.cuda.is_available():
                anchor = anchor.cuda()
                positive = positive.cuda()
                negative = negative.cuda()

            out1 = model(anchor)
            out2 = model(positive)
            out3 = model(negative)

            pos_dist = sk_pairwise.paired_distances(out1.cpu(), out2.cpu())
            neg_dist = sk_pairwise.paired_distances(out1.cpu(), out3.cpu())

            # Accuracy metric based loss's margin distance
            num_right += (pos_dist + margin <= neg_dist).sum().item()
            total_num += pos_dist.shape[0]
            
        return num_right / total_num


if __name__ == "__main__":
    random.seed(42)

    combinations = list(itertools.product(*config.TARGET_COMBINATIONS))
    random.shuffle(combinations)
    combinations = combinations[:_NUM_COMBINATIONS]

    model = target_typer.TargetTyper(
        num_classes=len(combinations),
        backbone="resnet18",
        use_cuda=torch.cuda.is_available()
    )

    if torch.cuda.is_available():
        model.cuda()

    classes = {
        "_".join([str(item) for item in name]): idx
        for idx, name in enumerate(itertools.product(*config.TARGET_COMBINATIONS))
    }

    dataset = datasets.TargetDataset(
        pathlib.Path("data_generation/data/combinations_train/images"),
        img_ext=config.IMAGE_EXT,
        img_width=90,
        img_height=90,
        classes=classes,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=65, pin_memory=True, shuffle=True
    )

    dataset = datasets.TargetDataset(
        pathlib.Path("data_generation/data/combinations_val/images"),
        img_ext=config.IMAGE_EXT,
        img_width=90,
        img_height=90,
        classes=classes,
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset, batch_size=65, pin_memory=True, shuffle=True
    )

    train(model, train_loader, eval_loader)
