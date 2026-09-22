"""Shared device, reproducibility and checkpoint helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, directory / "train_checkpoint.pth.tar")
    path = directory / ("train_model_%d.pth" % epoch)
    torch.save(model.state_dict(), path)
    return path


def load_weights(model, path, device):
    state = torch.load(str(path), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)


def latest_weights(directory):
    paths = list(Path(directory).glob("train_model_*.pth"))
    if not paths:
        raise FileNotFoundError("No checkpoint found in %s" % directory)
    return max(paths, key=lambda path: path.stat().st_mtime)

