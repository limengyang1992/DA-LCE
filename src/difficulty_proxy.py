"""
Difficulty Proxy Calculation Module
Implements early-stage difficulty proxy as described in the DA-LCE paper.
"""

import numpy as np
from typing import Union, Tuple
import torch


class DifficultyProxy:
    """
    Computes difficulty proxy from learning curves.

    The difficulty proxy consists of three components:
    1. Rate of Progress (ϕ_prog): Average improvement per epoch
    2. Non-Linearity (ϕ_nonlin): Deviation from linear trajectory
    3. Volatility (ϕ_vol): Standard deviation of first-order differences
    """

    def __init__(self, mean: np.ndarray = None, std: np.ndarray = None):
        """
        Args:
            mean: Mean values for standardization (shape: [3])
            std: Standard deviation values for standardization (shape: [3])
        """
        self.mean = mean
        self.std = std

    def compute_progress(self, curve: np.ndarray) -> float:
        """
        Compute rate of progress: (y_T - y_1) / T

        Args:
            curve: Learning curve array of shape [T]

        Returns:
            Progress rate
        """
        T = len(curve)
        if T < 2:
            return 0.0
        return (curve[-1] - curve[0]) / T

    def compute_nonlinearity(self, curve: np.ndarray) -> float:
        """
        Compute non-linearity: MSE relative to linear interpolation

        Args:
            curve: Learning curve array of shape [T]

        Returns:
            Non-linearity score
        """
        T = len(curve)
        if T < 2:
            return 0.0

        # Create linear interpolation from start to end
        linear = np.linspace(curve[0], curve[-1], T)

        # Compute mean squared error
        mse = np.mean((curve - linear) ** 2)
        return mse

    def compute_volatility(self, curve: np.ndarray) -> float:
        """
        Compute volatility: standard deviation of first-order differences

        Args:
            curve: Learning curve array of shape [T]

        Returns:
            Volatility score
        """
        if len(curve) < 2:
            return 0.0

        # First-order differences
        diffs = np.diff(curve)

        # Standard deviation
        return np.std(diffs)

    def compute(self, curve: Union[np.ndarray, torch.Tensor],
                standardize: bool = True) -> np.ndarray:
        """
        Compute full difficulty proxy vector: [ϕ_prog, ϕ_nonlin, ϕ_vol]

        Args:
            curve: Learning curve of shape [T] or [T, 1]
            standardize: Whether to standardize the proxy vector

        Returns:
            Difficulty proxy vector of shape [3]
        """
        # Convert to numpy if needed
        if isinstance(curve, torch.Tensor):
            curve = curve.cpu().numpy()

        # Flatten if needed
        if curve.ndim > 1:
            curve = curve.flatten()

        # Compute components
        prog = self.compute_progress(curve)
        nonlin = self.compute_nonlinearity(curve)
        vol = self.compute_volatility(curve)

        # Create proxy vector
        proxy = np.array([prog, nonlin, vol], dtype=np.float32)

        # Standardize if requested and statistics are available
        if standardize and self.mean is not None and self.std is not None:
            proxy = (proxy - self.mean) / (self.std + 1e-8)

        return proxy

    def fit_standardization(self, curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit standardization parameters from a corpus of curves.

        Args:
            curves: Array of curves, shape [N, T] or list of variable-length arrays

        Returns:
            mean, std: Standardization parameters, each of shape [3]
        """
        proxies = []

        if isinstance(curves, (list, tuple)):
            # Variable length curves
            for curve in curves:
                proxy = self.compute(curve, standardize=False)
                proxies.append(proxy)
        else:
            # Fixed length curves
            for i in range(len(curves)):
                proxy = self.compute(curves[i], standardize=False)
                proxies.append(proxy)

        proxies = np.array(proxies)

        # Compute statistics
        self.mean = np.mean(proxies, axis=0)
        self.std = np.std(proxies, axis=0)

        return self.mean, self.std

    def save_statistics(self, path: str):
        """Save standardization statistics to file."""
        np.savez(path, mean=self.mean, std=self.std)

    def load_statistics(self, path: str):
        """Load standardization statistics from file."""
        data = np.load(path)
        self.mean = data['mean']
        self.std = data['std']


def compute_difficulty_proxy(curve: Union[np.ndarray, torch.Tensor],
                             mean: np.ndarray = None,
                             std: np.ndarray = None) -> np.ndarray:
    """
    Convenience function to compute difficulty proxy.

    Args:
        curve: Learning curve
        mean: Mean for standardization
        std: Std for standardization

    Returns:
        Difficulty proxy vector [3]
    """
    proxy_computer = DifficultyProxy(mean=mean, std=std)
    return proxy_computer.compute(curve, standardize=(mean is not None))
