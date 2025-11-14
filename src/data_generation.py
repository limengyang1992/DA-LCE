"""
Data Generation Pipeline
Handles synthetic data generation using conditional DDPM and GMM.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from sklearn.mixture import GaussianMixture
from .difficulty_proxy import DifficultyProxy


class DynamicsGMM:
    """
    Gaussian Mixture Model for learning curve dynamics distribution.
    """

    def __init__(self, n_components: int = 5):
        """
        Args:
            n_components: Number of GMM components
        """
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=n_components,
                                    covariance_type='full',
                                    random_state=42)
        self.fitted = False

    def fit(self, curves: List[np.ndarray]) -> None:
        """
        Fit GMM to dynamics vectors from real curves.

        Args:
            curves: List of learning curves (variable length)
        """
        # Compute full-trajectory dynamics for each curve
        proxy_computer = DifficultyProxy()
        dynamics_vectors = []

        for curve in curves:
            dynamics = proxy_computer.compute(curve, standardize=False)
            dynamics_vectors.append(dynamics)

        dynamics_vectors = np.array(dynamics_vectors)

        # Fit GMM
        self.gmm.fit(dynamics_vectors)
        self.fitted = True

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample target dynamics vectors from the fitted GMM.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Sampled dynamics vectors [n_samples, 3]
        """
        if not self.fitted:
            raise ValueError("GMM must be fitted before sampling")

        samples, _ = self.gmm.sample(n_samples)
        return samples.astype(np.float32)

    def save(self, path: str):
        """Save GMM model."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.gmm, f)

    def load(self, path: str):
        """Load GMM model."""
        import pickle
        with open(path, 'rb') as f:
            self.gmm = pickle.load(f)
        self.fitted = True


class SyntheticDataGenerator:
    """
    Generate synthetic learning curves using conditional DDPM.
    """

    def __init__(self,
                 ddpm_model,
                 dynamics_gmm: DynamicsGMM,
                 curve_length: int = 100,
                 device: str = 'cuda'):
        """
        Args:
            ddpm_model: Trained conditional DDPM model
            dynamics_gmm: Fitted GMM for dynamics distribution
            curve_length: Length of generated curves
            device: Device to run on
        """
        self.ddpm = ddpm_model
        self.gmm = dynamics_gmm
        self.curve_length = curve_length
        self.device = device

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of synthetic learning curves.

        Args:
            batch_size: Number of curves to generate

        Returns:
            curves: Generated curves [batch_size, curve_length]
            dynamics: Corresponding dynamics vectors [batch_size, 3]
        """
        # Sample target dynamics from GMM
        target_dynamics = self.gmm.sample(batch_size)
        target_dynamics_tensor = torch.from_numpy(target_dynamics).to(self.device)

        # Generate curves using DDPM
        shape = (batch_size, 1, self.curve_length)
        generated_curves = self.ddpm.sample(shape, target_dynamics_tensor)

        # Squeeze to [batch_size, curve_length]
        generated_curves = generated_curves.squeeze(1)

        return generated_curves, target_dynamics_tensor

    def generate_dataset(self, num_curves: int, batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a full synthetic dataset.

        Args:
            num_curves: Total number of curves to generate
            batch_size: Batch size for generation

        Returns:
            all_curves: All generated curves [num_curves, curve_length]
            all_dynamics: All dynamics vectors [num_curves, 3]
        """
        all_curves = []
        all_dynamics = []

        num_batches = (num_curves + batch_size - 1) // batch_size

        for i in range(num_batches):
            current_batch_size = min(batch_size, num_curves - i * batch_size)

            curves, dynamics = self.generate_batch(current_batch_size)

            all_curves.append(curves.cpu().numpy())
            all_dynamics.append(dynamics.cpu().numpy())

        all_curves = np.concatenate(all_curves, axis=0)
        all_dynamics = np.concatenate(all_dynamics, axis=0)

        return all_curves, all_dynamics


class ParametricCurveGenerator:
    """
    Generate learning curves using parametric functions (baseline).
    """

    @staticmethod
    def power_law(epochs: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Power law: y = a - b * x^(-c)"""
        return a - b * np.power(epochs + 1, -c)

    @staticmethod
    def exponential(epochs: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Exponential: y = a - b * exp(-c * x)"""
        return a - b * np.exp(-c * epochs)

    @staticmethod
    def logistic(epochs: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Logistic: y = a / (1 + b * exp(-c * x))"""
        return a / (1 + b * np.exp(-c * epochs))

    @staticmethod
    def log_log_linear(epochs: np.ndarray, a: float, b: float) -> np.ndarray:
        """Log-log linear: y = a - b / log(x + e)"""
        return a - b / np.log(epochs + np.e)

    def generate_curve(self,
                       curve_type: str,
                       length: int,
                       noise_level: float = 0.01) -> np.ndarray:
        """
        Generate a parametric learning curve.

        Args:
            curve_type: Type of curve ('power', 'exp', 'logistic', 'loglog')
            length: Number of epochs
            noise_level: Gaussian noise level

        Returns:
            Generated curve [length]
        """
        epochs = np.arange(length)

        # Random parameters
        if curve_type == 'power':
            a = np.random.uniform(0.8, 0.95)
            b = np.random.uniform(0.3, 0.8)
            c = np.random.uniform(0.3, 0.8)
            curve = self.power_law(epochs, a, b, c)

        elif curve_type == 'exp':
            a = np.random.uniform(0.8, 0.95)
            b = np.random.uniform(0.3, 0.8)
            c = np.random.uniform(0.01, 0.1)
            curve = self.exponential(epochs, a, b, c)

        elif curve_type == 'logistic':
            a = np.random.uniform(0.8, 0.95)
            b = np.random.uniform(5, 15)
            c = np.random.uniform(0.05, 0.2)
            curve = self.logistic(epochs, a, b, c)

        elif curve_type == 'loglog':
            a = np.random.uniform(0.8, 0.95)
            b = np.random.uniform(0.1, 0.3)
            curve = self.log_log_linear(epochs, a, b)

        else:
            raise ValueError(f"Unknown curve type: {curve_type}")

        # Add noise
        noise = np.random.normal(0, noise_level, length)
        curve = curve + noise

        # Clip to valid range
        curve = np.clip(curve, 0.0, 1.0)

        return curve.astype(np.float32)

    def generate_dataset(self,
                        num_curves: int,
                        length: int,
                        noise_level: float = 0.01) -> np.ndarray:
        """
        Generate a dataset of parametric curves.

        Args:
            num_curves: Number of curves
            length: Curve length
            noise_level: Noise level

        Returns:
            Dataset of curves [num_curves, length]
        """
        curves = []
        curve_types = ['power', 'exp', 'logistic', 'loglog']

        for _ in range(num_curves):
            curve_type = np.random.choice(curve_types)
            curve = self.generate_curve(curve_type, length, noise_level)
            curves.append(curve)

        return np.array(curves)
