import os
from typing import Union, Optional

import torch
from torch import optim

from Model import Model


def save_checkpoint(
        model: Model,
        optimizer: optim.AdamW,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau,
        epoch: Union[int, str]
):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, f"checkpoints/{epoch}.pth")


def load_checkpoint(
        model: Model,
        optimizer: Optional[optim.AdamW] = None,
        scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None
) -> int:
    latest_epoch = 0

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    for checkpoint in os.listdir("checkpoints"):
        checkpoint_epoch = int(checkpoint.split(".")[0])
        if checkpoint_epoch > latest_epoch:
            latest_epoch = checkpoint_epoch

    if latest_epoch > 0:
        checkpoint = torch.load(f"checkpoints/{latest_epoch}.pth")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from epoch {latest_epoch}")
        return latest_epoch + 1

    return 0
