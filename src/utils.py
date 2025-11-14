"""
Utility Functions
"""

import torch
import numpy as np
import random
import os
from typing import Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> str:
    """Get computing device."""
    if device is not None:
        return device

    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_results(results: dict, path: str):
    """Save results to file."""
    import json

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {path}")


def load_results(path: str) -> dict:
    """Load results from file."""
    import json

    with open(path, 'r') as f:
        results = json.load(f)

    return results


def normalize_curve(curve: np.ndarray,
                    min_val: float = 0.0,
                    max_val: float = 1.0) -> np.ndarray:
    """
    Normalize a learning curve to [min_val, max_val].

    Args:
        curve: Learning curve
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Normalized curve
    """
    curve_min = curve.min()
    curve_max = curve.max()

    if curve_max == curve_min:
        return np.full_like(curve, (min_val + max_val) / 2)

    normalized = (curve - curve_min) / (curve_max - curve_min)
    normalized = normalized * (max_val - min_val) + min_val

    return normalized


def create_train_val_split(data: np.ndarray,
                           val_ratio: float = 0.2,
                           shuffle: bool = True,
                           seed: int = 42):
    """
    Split data into train and validation sets.

    Args:
        data: Data array
        val_ratio: Validation ratio
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        train_data, val_data
    """
    n = len(data)
    indices = np.arange(n)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    val_size = int(n * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return data[train_indices], data[val_indices]
