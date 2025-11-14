"""
Conditional Denoising Diffusion Probabilistic Model (DDPM) for Learning Curves
Implements the conditional diffusion model for synthetic data generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 cond_emb_dim: int, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Condition embedding projection (difficulty proxy)
        self.cond_mlp = nn.Linear(cond_emb_dim, out_channels)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.dropout = nn.Dropout(dropout)

        # Shortcut connection
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor,
                cond_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, L]
            time_emb: Time embedding [B, time_emb_dim]
            cond_emb: Condition embedding [B, cond_emb_dim]

        Returns:
            Output tensor [B, C, L]
        """
        h = self.conv1(x)
        h = self.norm1(h)

        # Add time and condition embeddings
        time_out = self.time_mlp(time_emb)[:, :, None]
        cond_out = self.cond_mlp(cond_emb)[:, :, None]
        h = h + time_out + cond_out

        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return F.relu(h + self.shortcut(x))


class UNet1D(nn.Module):
    """1D U-Net for denoising learning curves."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 channels: int = 64, time_emb_dim: int = 128,
                 cond_dim: int = 3, cond_emb_dim: int = 64,
                 num_res_blocks: int = 2):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Condition (difficulty proxy) embedding
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_emb_dim),
            nn.ReLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim),
            nn.ReLU()
        )

        # Initial convolution
        self.init_conv = nn.Conv1d(in_channels, channels, kernel_size=3, padding=1)

        # Encoder (downsampling)
        self.down1 = nn.ModuleList([
            ResidualBlock(channels, channels, time_emb_dim, cond_emb_dim)
            for _ in range(num_res_blocks)
        ])
        self.down_sample1 = nn.Conv1d(channels, channels * 2, kernel_size=3,
                                       stride=2, padding=1)

        self.down2 = nn.ModuleList([
            ResidualBlock(channels * 2, channels * 2, time_emb_dim, cond_emb_dim)
            for _ in range(num_res_blocks)
        ])
        self.down_sample2 = nn.Conv1d(channels * 2, channels * 4, kernel_size=3,
                                       stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(channels * 4, channels * 4, time_emb_dim, cond_emb_dim)
            for _ in range(num_res_blocks)
        ])

        # Decoder (upsampling)
        self.up_sample1 = nn.ConvTranspose1d(channels * 4, channels * 2,
                                              kernel_size=4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResidualBlock(channels * 4, channels * 2, time_emb_dim, cond_emb_dim)
            for _ in range(num_res_blocks)
        ])

        self.up_sample2 = nn.ConvTranspose1d(channels * 2, channels,
                                              kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResidualBlock(channels * 2, channels, time_emb_dim, cond_emb_dim)
            for _ in range(num_res_blocks)
        ])

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            nn.Conv1d(channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy input [B, 1, L]
            time: Timestep [B]
            condition: Difficulty proxy [B, 3]

        Returns:
            Predicted noise [B, 1, L]
        """
        # Embeddings
        time_emb = self.time_mlp(time)
        cond_emb = self.cond_mlp(condition)

        # Initial conv
        x = self.init_conv(x)
        residual_inputs = [x]

        # Encoder
        for block in self.down1:
            x = block(x, time_emb, cond_emb)
            residual_inputs.append(x)
        x = self.down_sample1(x)

        for block in self.down2:
            x = block(x, time_emb, cond_emb)
            residual_inputs.append(x)
        x = self.down_sample2(x)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x, time_emb, cond_emb)

        # Decoder with skip connections
        x = self.up_sample1(x)
        x = torch.cat([x, residual_inputs.pop()], dim=1)
        for block in self.up1:
            x = block(x, time_emb, cond_emb)

        x = self.up_sample2(x)
        x = torch.cat([x, residual_inputs.pop()], dim=1)
        for block in self.up2:
            x = block(x, time_emb, cond_emb)

        return self.final_conv(x)


class ConditionalDDPM:
    """Conditional Denoising Diffusion Probabilistic Model."""

    def __init__(self, model: nn.Module, timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str = 'cuda'):
        """
        Args:
            model: UNet1D model
            timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            device: Device to run on
        """
        self.model = model
        self.timesteps = timesteps
        self.device = device

        # Define beta schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / \
                                  (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)

        Args:
            x_start: Clean data [B, 1, L]
            t: Timestep [B]
            noise: Optional noise tensor

        Returns:
            Noisy data [B, 1, L]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start: torch.Tensor, condition: torch.Tensor,
                 t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            x_start: Clean data [B, 1, L]
            condition: Difficulty proxy [B, 3]
            t: Timestep [B]
            noise: Optional noise tensor

        Returns:
            MSE loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, condition)

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, condition: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion step: p(x_{t-1} | x_t)

        Args:
            x: Noisy data at timestep t [B, 1, L]
            t: Current timestep (int)
            condition: Difficulty proxy [B, 3]

        Returns:
            Denoised data at timestep t-1
        """
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]

        # Predict noise
        t_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        predicted_noise = self.model(x, t_tensor, condition)

        # Compute mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, shape: Tuple[int, int, int],
               condition: torch.Tensor) -> torch.Tensor:
        """
        Generate samples from noise.

        Args:
            shape: Output shape [B, 1, L]
            condition: Difficulty proxy [B, 3]

        Returns:
            Generated learning curves
        """
        # Start from pure noise
        x = torch.randn(shape, device=self.device)

        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, condition)

        return x
