"""Utility helpers shared across the project."""
import json
import os
import random

import numpy as np
import torch
import yaml


def load_config(path):
    """Load a YAML config file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Fix random seeds so that experiments are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    """Use the GPU if one is available (e.g. on Google Colab), otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
