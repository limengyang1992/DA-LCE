"""
Conditional Difficulty-aware Prior-data Fitted Network (CD-PFN)
Transformer-based model for learning curve extrapolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, L, D]
        Returns:
            Positional encoding [L, D]
        """
        return self.pe[:x.size(1)]


class CDPFN(nn.Module):
    """
    Conditional Difficulty-aware Prior-data Fitted Network.

    This model uses a Transformer architecture to perform learning curve
    extrapolation conditioned on difficulty proxy.
    """

    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 num_bins: int = 100,
                 cond_dim: int = 3,
                 max_epochs: int = 200):
        """
        Args:
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            num_bins: Number of bins for discretized output
            cond_dim: Dimension of difficulty proxy (default 3)
            max_epochs: Maximum number of epochs in sequences
        """
        super().__init__()

        self.d_model = d_model
        self.num_bins = num_bins

        # Input embeddings for (epoch, performance) pairs
        self.epoch_embedding = nn.Linear(1, d_model // 2)
        self.performance_embedding = nn.Linear(1, d_model // 2)

        # Difficulty proxy embedding (global conditioning)
        self.difficulty_embedding = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_epochs + 1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output head for prediction
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_bins)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                epochs: torch.Tensor,
                performances: torch.Tensor,
                difficulty_proxy: torch.Tensor,
                target_epoch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            epochs: Observed epoch indices [B, T]
            performances: Observed performance values [B, T]
            difficulty_proxy: Difficulty proxy vector [B, 3]
            target_epoch: Target epoch for prediction [B] (optional)

        Returns:
            Predicted distribution over bins [B, num_bins]
        """
        B, T = epochs.shape

        # Embed epochs and performances
        epoch_emb = self.epoch_embedding(epochs.unsqueeze(-1))  # [B, T, d_model/2]
        perf_emb = self.performance_embedding(performances.unsqueeze(-1))  # [B, T, d_model/2]

        # Concatenate to form input embeddings
        input_emb = torch.cat([epoch_emb, perf_emb], dim=-1)  # [B, T, d_model]

        # Create difficulty token (prepended to sequence)
        difficulty_emb = self.difficulty_embedding(difficulty_proxy)  # [B, d_model]
        difficulty_token = difficulty_emb.unsqueeze(1)  # [B, 1, d_model]

        # Combine difficulty token with input sequence
        sequence = torch.cat([difficulty_token, input_emb], dim=1)  # [B, T+1, d_model]

        # Add positional encoding
        pos_enc = self.pos_encoder(sequence)  # [T+1, d_model]
        sequence = sequence + pos_enc.unsqueeze(0)  # [B, T+1, d_model]

        # Apply transformer
        encoded = self.transformer_encoder(sequence)  # [B, T+1, d_model]

        # Use the last token for prediction (or difficulty token for global prediction)
        # Here we use the last observed point
        output = encoded[:, -1, :]  # [B, d_model]

        # Predict distribution over bins
        logits = self.output_head(output)  # [B, num_bins]

        return logits

    def predict_distribution(self,
                            epochs: torch.Tensor,
                            performances: torch.Tensor,
                            difficulty_proxy: torch.Tensor) -> torch.Tensor:
        """
        Get probability distribution over performance bins.

        Args:
            epochs: Observed epochs [B, T]
            performances: Observed performances [B, T]
            difficulty_proxy: Difficulty proxy [B, 3]

        Returns:
            Probability distribution [B, num_bins]
        """
        logits = self(epochs, performances, difficulty_proxy)
        return F.softmax(logits, dim=-1)

    def predict_mean(self,
                     epochs: torch.Tensor,
                     performances: torch.Tensor,
                     difficulty_proxy: torch.Tensor,
                     bin_edges: torch.Tensor) -> torch.Tensor:
        """
        Predict expected value.

        Args:
            epochs: Observed epochs [B, T]
            performances: Observed performances [B, T]
            difficulty_proxy: Difficulty proxy [B, 3]
            bin_edges: Bin edges for discretization [num_bins + 1]

        Returns:
            Expected performance value [B]
        """
        probs = self.predict_distribution(epochs, performances, difficulty_proxy)

        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers = bin_centers.to(probs.device)

        # Expected value
        expected = (probs * bin_centers).sum(dim=-1)

        return expected

    def compute_loss(self,
                     epochs: torch.Tensor,
                     performances: torch.Tensor,
                     difficulty_proxy: torch.Tensor,
                     target_values: torch.Tensor,
                     bin_edges: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute training loss.

        Args:
            epochs: Observed epochs [B, T]
            performances: Observed performances [B, T]
            difficulty_proxy: Difficulty proxy [B, 3]
            target_values: True target performance values [B]
            bin_edges: Bin edges [num_bins + 1]

        Returns:
            loss: Cross-entropy loss
            accuracy: Prediction accuracy
        """
        # Get logits
        logits = self(epochs, performances, difficulty_proxy)

        # Discretize target values
        target_bins = torch.searchsorted(bin_edges, target_values) - 1
        target_bins = torch.clamp(target_bins, 0, self.num_bins - 1)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, target_bins)

        # Accuracy
        predicted_bins = torch.argmax(logits, dim=-1)
        accuracy = (predicted_bins == target_bins).float().mean()

        return loss, accuracy


class PerformanceDiscretizer:
    """Helper class for discretizing performance values into bins."""

    def __init__(self, num_bins: int = 100, min_val: float = 0.0, max_val: float = 1.0):
        """
        Args:
            num_bins: Number of bins
            min_val: Minimum performance value
            max_val: Maximum performance value
        """
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.bin_edges = torch.linspace(min_val, max_val, num_bins + 1)

    def discretize(self, values: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous values to bin indices.

        Args:
            values: Continuous values [...]

        Returns:
            Bin indices [...]
        """
        bins = torch.searchsorted(self.bin_edges.to(values.device), values) - 1
        return torch.clamp(bins, 0, self.num_bins - 1)

    def continuize(self, bins: torch.Tensor) -> torch.Tensor:
        """
        Convert bin indices to continuous values (bin centers).

        Args:
            bins: Bin indices [...]

        Returns:
            Continuous values [...]
        """
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        return bin_centers[bins.long()]

    def get_bin_edges(self) -> torch.Tensor:
        """Get bin edges tensor."""
        return self.bin_edges
