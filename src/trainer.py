"""
Training Pipeline for DA-LCE
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm
import os

from .difficulty_proxy import DifficultyProxy
from .cd_pfn import CDPFN, PerformanceDiscretizer


class LearningCurveDataset(Dataset):
    """Dataset for learning curves."""

    def __init__(self,
                 curves: np.ndarray,
                 difficulty_proxy_calculator: DifficultyProxy,
                 min_history: int = 10,
                 max_history: int = 50):
        """
        Args:
            curves: Array of learning curves [N, max_length]
            difficulty_proxy_calculator: Calculator for difficulty proxy
            min_history: Minimum history length
            max_history: Maximum history length
        """
        self.curves = curves
        self.proxy_calc = difficulty_proxy_calculator
        self.min_history = min_history
        self.max_history = max_history

    def __len__(self) -> int:
        return len(self.curves)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Sample a training instance:
        - Random cutoff point T in [min_history, max_history]
        - Random target epoch t' > T
        - Return partial curve, difficulty proxy, and target value
        """
        curve = self.curves[idx]
        curve_length = len(curve)

        # Random cutoff point
        T = np.random.randint(self.min_history,
                             min(self.max_history, curve_length - 1))

        # Random target epoch (after cutoff)
        target_epoch = np.random.randint(T + 1, curve_length)

        # Partial curve
        partial_curve = curve[:T]

        # Compute difficulty proxy from partial curve
        difficulty_proxy = self.proxy_calc.compute(partial_curve)

        # Prepare data
        epochs = np.arange(T, dtype=np.float32)
        performances = partial_curve.astype(np.float32)
        target_value = curve[target_epoch]

        return {
            'epochs': torch.from_numpy(epochs),
            'performances': torch.from_numpy(performances),
            'difficulty_proxy': torch.from_numpy(difficulty_proxy),
            'target_value': torch.tensor(target_value, dtype=torch.float32),
            'target_epoch': torch.tensor(target_epoch, dtype=torch.long)
        }


class CDPFNTrainer:
    """Trainer for CD-PFN model."""

    def __init__(self,
                 model: CDPFN,
                 discretizer: PerformanceDiscretizer,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        """
        Args:
            model: CD-PFN model
            discretizer: Performance discretizer
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        self.model = model.to(device)
        self.discretizer = discretizer
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.bin_edges = discretizer.get_bin_edges().to(device)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            # Move to device
            epochs = batch['epochs'].to(self.device)
            performances = batch['performances'].to(self.device)
            difficulty_proxy = batch['difficulty_proxy'].to(self.device)
            target_values = batch['target_value'].to(self.device)

            # Forward pass
            loss, accuracy = self.model.compute_loss(
                epochs, performances, difficulty_proxy,
                target_values, self.bin_edges
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy.item():.4f}'
            })

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                epochs = batch['epochs'].to(self.device)
                performances = batch['performances'].to(self.device)
                difficulty_proxy = batch['difficulty_proxy'].to(self.device)
                target_values = batch['target_value'].to(self.device)

                # Compute loss
                loss, accuracy = self.model.compute_loss(
                    epochs, performances, difficulty_proxy,
                    target_values, self.bin_edges
                )

                # Compute MAE
                predictions = self.model.predict_mean(
                    epochs, performances, difficulty_proxy, self.bin_edges
                )
                mae = torch.abs(predictions - target_values).mean()

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_mae += mae.item()
                num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'mae': total_mae / num_batches
        }

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              save_dir: str = 'checkpoints') -> List[Dict[str, float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            save_dir: Directory to save checkpoints

        Returns:
            Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        history = []

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")

            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}, "
                      f"Val MAE: {val_metrics['mae']:.4f}")

                # Save best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    self.save_checkpoint(
                        os.path.join(save_dir, 'best_model.pt'),
                        epoch, val_metrics
                    )

                history.append({**train_metrics, **{'val_' + k: v for k, v in val_metrics.items()}})
            else:
                history.append(train_metrics)

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt'),
                    epoch, train_metrics
                )

        return history

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['metrics']
