"""
Example script for training CD-PFN model.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.difficulty_proxy import DifficultyProxy
from src.cd_pfn import CDPFN, PerformanceDiscretizer
from src.trainer import LearningCurveDataset, CDPFNTrainer
from src.data_generation import ParametricCurveGenerator
from src.utils import set_seed, get_device, create_train_val_split


def main():
    # Set seed for reproducibility
    set_seed(42)

    # Configuration
    config = {
        'num_curves': 10000,
        'curve_length': 100,
        'num_bins': 100,
        'batch_size': 128,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 6,
        'min_history': 10,
        'max_history': 50
    }

    device = get_device()
    print(f"Using device: {device}")

    # Generate synthetic training data using parametric curves
    print("Generating synthetic data...")
    generator = ParametricCurveGenerator()
    curves = generator.generate_dataset(
        num_curves=config['num_curves'],
        length=config['curve_length'],
        noise_level=0.01
    )

    print(f"Generated {len(curves)} curves")

    # Split into train/val
    train_curves, val_curves = create_train_val_split(curves, val_ratio=0.2)
    print(f"Train: {len(train_curves)}, Val: {len(val_curves)}")

    # Initialize difficulty proxy calculator
    print("Fitting difficulty proxy statistics...")
    proxy_calc = DifficultyProxy()
    proxy_calc.fit_standardization(train_curves)
    proxy_calc.save_statistics('difficulty_stats.npz')

    # Create datasets
    train_dataset = LearningCurveDataset(
        train_curves,
        proxy_calc,
        min_history=config['min_history'],
        max_history=config['max_history']
    )

    val_dataset = LearningCurveDataset(
        val_curves,
        proxy_calc,
        min_history=config['min_history'],
        max_history=config['max_history']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    print("Initializing CD-PFN model...")
    model = CDPFN(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        num_bins=config['num_bins'],
        max_epochs=config['curve_length']
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize discretizer
    discretizer = PerformanceDiscretizer(
        num_bins=config['num_bins'],
        min_val=0.0,
        max_val=1.0
    )

    # Initialize trainer
    trainer = CDPFNTrainer(
        model=model,
        discretizer=discretizer,
        device=device,
        learning_rate=config['learning_rate']
    )

    # Train
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        save_dir='checkpoints'
    )

    print("Training completed!")

    # Save final model
    trainer.save_checkpoint('checkpoints/final_model.pt', config['num_epochs'], {})

    # Save training history
    import json
    with open('checkpoints/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("Model and history saved!")


if __name__ == '__main__':
    main()
