"""
Example script for training conditional DDPM.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

from src.ddpm import UNet1D, ConditionalDDPM
from src.difficulty_proxy import DifficultyProxy
from src.data_generation import ParametricCurveGenerator, DynamicsGMM
from src.utils import set_seed, get_device


class CurveDataset(Dataset):
    """Dataset of curves with difficulty proxies."""

    def __init__(self, curves, difficulty_proxies):
        self.curves = curves
        self.difficulty_proxies = difficulty_proxies

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, idx):
        return {
            'curve': torch.from_numpy(self.curves[idx]).unsqueeze(0).float(),
            'difficulty': torch.from_numpy(self.difficulty_proxies[idx]).float()
        }


def train_ddpm_epoch(ddpm, dataloader, optimizer, device):
    """Train DDPM for one epoch."""
    ddpm.model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training DDPM"):
        curves = batch['curve'].to(device)
        difficulties = batch['difficulty'].to(device)

        # Random timesteps
        t = torch.randint(0, ddpm.timesteps, (curves.shape[0],), device=device).long()

        # Compute loss
        loss = ddpm.p_losses(curves, difficulties, t)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    set_seed(42)

    # Configuration
    config = {
        'num_curves': 20000,
        'curve_length': 100,
        'batch_size': 64,
        'num_epochs': 100,
        'learning_rate': 2e-4,
        'timesteps': 1000
    }

    device = get_device()
    print(f"Using device: {device}")

    # Generate real curves
    print("Generating real curves...")
    generator = ParametricCurveGenerator()
    real_curves = generator.generate_dataset(
        num_curves=config['num_curves'],
        length=config['curve_length']
    )

    # Compute difficulty proxies
    print("Computing difficulty proxies...")
    proxy_calc = DifficultyProxy()
    proxy_calc.fit_standardization(real_curves)

    difficulty_proxies = []
    for curve in real_curves:
        proxy = proxy_calc.compute(curve, standardize=False)
        difficulty_proxies.append(proxy)
    difficulty_proxies = np.array(difficulty_proxies)

    # Fit GMM to dynamics distribution
    print("Fitting GMM to dynamics distribution...")
    gmm = DynamicsGMM(n_components=5)
    gmm.fit(real_curves)
    gmm.save('dynamics_gmm.pkl')

    # Create dataset
    dataset = CurveDataset(real_curves, difficulty_proxies)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Initialize DDPM model
    print("Initializing DDPM...")
    unet = UNet1D(
        in_channels=1,
        out_channels=1,
        channels=64,
        time_emb_dim=128,
        cond_dim=3,
        cond_emb_dim=64
    )

    ddpm = ConditionalDDPM(
        model=unet,
        timesteps=config['timesteps'],
        device=device
    )

    print(f"DDPM parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'])

    # Training loop
    print("Starting training...")
    os.makedirs('checkpoints_ddpm', exist_ok=True)

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        loss = train_ddpm_epoch(ddpm, dataloader, optimizer, device)
        print(f"Loss: {loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints_ddpm/ddpm_epoch_{epoch + 1}.pt')

    # Save final model
    torch.save(unet.state_dict(), 'checkpoints_ddpm/ddpm_final.pt')
    print("DDPM training completed!")


if __name__ == '__main__':
    main()
