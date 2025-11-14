"""
Test script to verify installation and basic functionality.
"""

import sys
sys.path.append('..')

import torch
import numpy as np


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.difficulty_proxy import DifficultyProxy, compute_difficulty_proxy
        from src.ddpm import UNet1D, ConditionalDDPM
        from src.cd_pfn import CDPFN, PerformanceDiscretizer
        from src.data_generation import (ParametricCurveGenerator,
                                         DynamicsGMM,
                                         SyntheticDataGenerator)
        from src.trainer import LearningCurveDataset, CDPFNTrainer
        from src.evaluator import LCEvaluator, EarlyStoppingSimulator
        from src.utils import set_seed, get_device

        print("✓ All imports successful")
        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_difficulty_proxy():
    """Test difficulty proxy computation."""
    print("\nTesting difficulty proxy...")

    try:
        from src.difficulty_proxy import DifficultyProxy

        # Create sample curve
        curve = np.linspace(0.1, 0.9, 50)

        # Compute proxy
        proxy_calc = DifficultyProxy()
        proxy = proxy_calc.compute(curve, standardize=False)

        assert proxy.shape == (3,), f"Expected shape (3,), got {proxy.shape}"
        assert np.all(np.isfinite(proxy)), "Proxy contains non-finite values"

        print(f"✓ Difficulty proxy computed: {proxy}")
        return True

    except Exception as e:
        print(f"✗ Difficulty proxy test failed: {e}")
        return False


def test_parametric_generation():
    """Test parametric curve generation."""
    print("\nTesting parametric curve generation...")

    try:
        from src.data_generation import ParametricCurveGenerator

        generator = ParametricCurveGenerator()

        # Test each curve type
        for curve_type in ['power', 'exp', 'logistic', 'loglog']:
            curve = generator.generate_curve(curve_type, length=100)
            assert curve.shape == (100,), f"Expected shape (100,), got {curve.shape}"
            assert np.all((curve >= 0) & (curve <= 1)), f"{curve_type} curve out of range"

        print("✓ Parametric curve generation successful")
        return True

    except Exception as e:
        print(f"✗ Parametric generation test failed: {e}")
        return False


def test_model_initialization():
    """Test model initialization."""
    print("\nTesting model initialization...")

    try:
        from src.cd_pfn import CDPFN
        from src.ddpm import UNet1D

        # Test CD-PFN
        model = CDPFN(d_model=128, nhead=4, num_layers=2)
        print(f"  CD-PFN parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test UNet1D
        unet = UNet1D(channels=32, num_res_blocks=1)
        print(f"  UNet1D parameters: {sum(p.numel() for p in unet.parameters()):,}")

        print("✓ Model initialization successful")
        return True

    except Exception as e:
        print(f"✗ Model initialization test failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass through models."""
    print("\nTesting forward pass...")

    try:
        from src.cd_pfn import CDPFN
        from src.ddpm import UNet1D, ConditionalDDPM

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Test CD-PFN forward
        model = CDPFN(d_model=128, nhead=4, num_layers=2).to(device)
        epochs = torch.arange(10, dtype=torch.float32).unsqueeze(0).to(device)
        performances = torch.rand(1, 10).to(device)
        difficulty = torch.rand(1, 3).to(device)

        output = model(epochs, performances, difficulty)
        assert output.shape == (1, 100), f"Expected shape (1, 100), got {output.shape}"

        print("  ✓ CD-PFN forward pass successful")

        # Test UNet1D forward
        unet = UNet1D(channels=32, num_res_blocks=1).to(device)
        x = torch.randn(2, 1, 100).to(device)
        t = torch.tensor([50, 100]).to(device)
        cond = torch.rand(2, 3).to(device)

        output = unet(x, t, cond)
        assert output.shape == (2, 1, 100), f"Expected shape (2, 1, 100), got {output.shape}"

        print("  ✓ UNet1D forward pass successful")
        print("✓ All forward passes successful")
        return True

    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_discretizer():
    """Test performance discretizer."""
    print("\nTesting performance discretizer...")

    try:
        from src.cd_pfn import PerformanceDiscretizer

        discretizer = PerformanceDiscretizer(num_bins=50, min_val=0.0, max_val=1.0)

        # Test discretization
        values = torch.tensor([0.0, 0.5, 1.0])
        bins = discretizer.discretize(values)

        assert bins.shape == values.shape, "Shape mismatch"
        assert torch.all((bins >= 0) & (bins < 50)), "Bins out of range"

        # Test continuization
        continuous = discretizer.continuize(bins)
        assert continuous.shape == bins.shape, "Shape mismatch"

        print("✓ Performance discretizer successful")
        return True

    except Exception as e:
        print(f"✗ Discretizer test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DA-PFN Installation Test")
    print("=" * 60)

    tests = [
        test_imports,
        test_difficulty_proxy,
        test_parametric_generation,
        test_model_initialization,
        test_forward_pass,
        test_discretizer,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)

    if all(results):
        print("\n✓ All tests passed! Installation is successful.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
