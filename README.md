# DA-PFN: Difficulty-Aware Prior-data Fitted Networks

PyTorch implementation of "Difficulty-Aware Learning Curve Extrapolation" (DA-LCE) framework.

## Overview

This repository contains a complete implementation of the DA-LCE framework for learning curve extrapolation, as described in the paper. The framework introduces:

1. **Early-stage Difficulty Proxy**: A self-contained method to quantify task difficulty from learning curve dynamics
2. **Conditional DDPM**: A diffusion model for generating high-fidelity synthetic learning curves
3. **CD-PFN**: A Transformer-based model that conditions predictions on task difficulty

## Key Features

- **Difficulty-Aware Extrapolation**: Adapts predictions based on intrinsic task difficulty
- **High-Fidelity Synthetic Data**: Uses conditional diffusion models for realistic curve generation
- **Transformer Architecture**: Leverages self-attention for robust extrapolation
- **Comprehensive Evaluation**: Includes MAPE, MAE, RMSE, NLL, and difficulty-stratified metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DA-PFN.git
cd DA-PFN

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
DA-PFN/
├── src/
│   ├── difficulty_proxy.py    # Difficulty proxy calculation
│   ├── ddpm.py                 # Conditional DDPM implementation
│   ├── cd_pfn.py              # CD-PFN model architecture
│   ├── data_generation.py     # Data generation pipeline
│   ├── trainer.py             # Training utilities
│   ├── evaluator.py           # Evaluation metrics
│   └── utils.py               # Helper functions
├── examples/
│   ├── train_cd_pfn.py        # Training script for CD-PFN
│   ├── train_ddpm.py          # Training script for DDPM
│   └── evaluate.py            # Evaluation script
├── configs/                    # Configuration files
├── data/                       # Data directory
└── README.md
```

## Quick Start

### 1. Train CD-PFN (using parametric curves)

```bash
cd examples
python train_cd_pfn.py
```

This will:
- Generate 10,000 synthetic learning curves using parametric functions
- Fit difficulty proxy standardization parameters
- Train the CD-PFN model
- Save checkpoints to `checkpoints/`

### 2. Train Conditional DDPM (optional, for high-fidelity generation)

```bash
cd examples
python train_ddpm.py
```

This will:
- Generate real curves and compute their dynamics
- Fit a GMM to the dynamics distribution
- Train the conditional DDPM
- Save the model to `checkpoints_ddpm/`

### 3. Evaluate the Model

```bash
cd examples
python evaluate.py
```

This will:
- Load the trained model
- Evaluate on test curves
- Compute overall and difficulty-stratified metrics
- Save results to `evaluation_results.json`

## Usage Example

```python
import torch
import numpy as np
from src.difficulty_proxy import DifficultyProxy
from src.cd_pfn import CDPFN, PerformanceDiscretizer

# Initialize model
model = CDPFN(d_model=256, nhead=8, num_layers=6, num_bins=100)
model.load_state_dict(torch.load('checkpoints/best_model.pt')['model_state_dict'])
model.eval()

# Prepare input (partial learning curve)
epochs = torch.arange(30).unsqueeze(0)  # First 30 epochs
performances = torch.tensor(your_curve[:30]).unsqueeze(0)

# Compute difficulty proxy
proxy_calc = DifficultyProxy()
proxy_calc.load_statistics('difficulty_stats.npz')
difficulty_proxy = proxy_calc.compute(your_curve[:30])
difficulty_proxy = torch.from_numpy(difficulty_proxy).unsqueeze(0)

# Predict
discretizer = PerformanceDiscretizer(num_bins=100)
bin_edges = discretizer.get_bin_edges()
prediction = model.predict_mean(epochs, performances, difficulty_proxy, bin_edges)

print(f"Predicted final performance: {prediction.item():.4f}")
```

## Difficulty Proxy Components

The difficulty proxy consists of three interpretable components:

1. **Rate of Progress (φ_prog)**: `(y_T - y_1) / T`
   - Measures average improvement per epoch
   - Higher values indicate easier tasks

2. **Non-Linearity (φ_nonlin)**: MSE from linear interpolation
   - Captures complexity of learning dynamics
   - Higher values indicate more complex learning phases

3. **Volatility (φ_vol)**: Standard deviation of first-order differences
   - Measures training stability
   - Higher values indicate noisy/unstable training

## Model Architecture

### CD-PFN Architecture
- **Input**: Epoch-performance pairs + difficulty proxy
- **Embedding**: Separate embeddings for epochs and performances
- **Conditioning**: Difficulty proxy embedded and prepended as special token
- **Transformer**: Multi-head self-attention with 6 layers
- **Output**: Categorical distribution over performance bins

### Conditional DDPM Architecture
- **Backbone**: 1D U-Net with residual blocks
- **Conditioning**: Difficulty proxy injected via adaptive normalization
- **Timesteps**: 1000 diffusion steps with linear beta schedule
- **Training**: Simplified DDPM objective (noise prediction)

## Training Configuration

Default hyperparameters:

```python
{
    'num_curves': 10000,        # Training curves
    'curve_length': 100,        # Epochs per curve
    'num_bins': 100,           # Discretization bins
    'batch_size': 128,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'd_model': 256,            # Transformer hidden dim
    'nhead': 8,                # Attention heads
    'num_layers': 6,           # Transformer layers
    'min_history': 10,         # Min observation length
    'max_history': 50          # Max observation length
}
```

## Evaluation Metrics

The framework supports multiple evaluation metrics:

- **MAPE**: Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **NLL**: Negative Log-Likelihood
- **Correlation**: Spearman correlation coefficient

Results are reported both overall and stratified by difficulty (Easy/Medium/Hard).

## Citation

If you use this code, please cite:

```bibtex
@article{li2025difficulty,
  title={Difficulty-Aware Learning Curve Extrapolation},
  author={Li, Mengyang and Zhao, Pinlong},
  journal={AAAI},
  year={2025}
}
```

## Paper Reference

The implementation is based on:
- **Paper**: "Difficulty-Aware Learning Curve Extrapolation"
- **Authors**: Mengyang Li, Pinlong Zhao
- **Conference**: AAAI 2026

## Key Components

### Difficulty Proxy (Section 3.1)
- Computes three rule-based indicators from early curve dynamics
- Self-contained, no external meta-features needed
- Standardized using statistics from reference corpus

### Conditional Data Generation (Section 3.2)
- GMM fitted to real curve dynamics
- DDPM conditioned on target dynamics
- Generates high-fidelity synthetic curves

### CD-PFN Model (Section 3.3)
- Transformer with explicit difficulty conditioning
- Global [DYNAMICS] token prepended to sequence
- Categorical output over discretized performance bins

## Advanced Usage

### Custom Data

To use your own learning curves:

```python
from src.data_generation import DynamicsGMM
from src.difficulty_proxy import DifficultyProxy

# Load your curves (list of numpy arrays)
your_curves = [...]

# Fit difficulty proxy statistics
proxy_calc = DifficultyProxy()
proxy_calc.fit_standardization(your_curves)
proxy_calc.save_statistics('your_difficulty_stats.npz')

# Fit GMM to dynamics
gmm = DynamicsGMM(n_components=5)
gmm.fit(your_curves)
gmm.save('your_dynamics_gmm.pkl')
```

### Generate Synthetic Data with DDPM

```python
from src.ddpm import UNet1D, ConditionalDDPM
from src.data_generation import SyntheticDataGenerator, DynamicsGMM

# Load trained DDPM
unet = UNet1D()
unet.load_state_dict(torch.load('checkpoints_ddpm/ddpm_final.pt'))
ddpm = ConditionalDDPM(unet, timesteps=1000, device='cuda')

# Load GMM
gmm = DynamicsGMM()
gmm.load('dynamics_gmm.pkl')

# Generate synthetic curves
generator = SyntheticDataGenerator(ddpm, gmm, curve_length=100)
synthetic_curves, dynamics = generator.generate_dataset(num_curves=5000)
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.
