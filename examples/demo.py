"""
Simple demo script showing how to use DA-PFN for learning curve extrapolation.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.difficulty_proxy import DifficultyProxy
from src.data_generation import ParametricCurveGenerator


def demo_difficulty_proxy():
    """Demonstrate difficulty proxy computation."""
    print("=" * 60)
    print("DEMO 1: Computing Difficulty Proxy")
    print("=" * 60)

    # Generate sample curves
    generator = ParametricCurveGenerator()

    # Easy curve (fast convergence)
    easy_curve = generator.generate_curve('power', length=100, noise_level=0.005)

    # Hard curve (slow convergence, high volatility)
    hard_curve = generator.generate_curve('loglog', length=100, noise_level=0.03)

    # Compute difficulty proxies
    proxy_calc = DifficultyProxy()

    # Use early portion (first 30 epochs)
    early_portion = 30

    easy_proxy = proxy_calc.compute(easy_curve[:early_portion], standardize=False)
    hard_proxy = proxy_calc.compute(hard_curve[:early_portion], standardize=False)

    print("\nEasy Task Difficulty Proxy:")
    print(f"  Progress Rate:  {easy_proxy[0]:.6f} (higher is better)")
    print(f"  Non-linearity:  {easy_proxy[1]:.6f} (lower is simpler)")
    print(f"  Volatility:     {easy_proxy[2]:.6f} (lower is more stable)")

    print("\nHard Task Difficulty Proxy:")
    print(f"  Progress Rate:  {hard_proxy[0]:.6f} (higher is better)")
    print(f"  Non-linearity:  {hard_proxy[1]:.6f} (lower is simpler)")
    print(f"  Volatility:     {hard_proxy[2]:.6f} (lower is more stable)")

    # Visualize curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(easy_curve, label='Easy Task', linewidth=2)
    plt.axvline(x=early_portion, color='r', linestyle='--', label='Observation Point')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.title('Easy Task Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(hard_curve, label='Hard Task', linewidth=2)
    plt.axvline(x=early_portion, color='r', linestyle='--', label='Observation Point')
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.title('Hard Task Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('difficulty_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'difficulty_comparison.png'")


def demo_curve_generation():
    """Demonstrate parametric curve generation."""
    print("\n" + "=" * 60)
    print("DEMO 2: Generating Learning Curves")
    print("=" * 60)

    generator = ParametricCurveGenerator()

    curve_types = ['power', 'exp', 'logistic', 'loglog']

    plt.figure(figsize=(15, 4))

    for i, curve_type in enumerate(curve_types):
        # Generate multiple samples of each type
        curves = [generator.generate_curve(curve_type, length=100, noise_level=0.01)
                  for _ in range(5)]

        plt.subplot(1, 4, i + 1)
        for curve in curves:
            plt.plot(curve, alpha=0.6)
        plt.xlabel('Epoch')
        plt.ylabel('Performance')
        plt.title(f'{curve_type.capitalize()} Curves')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('curve_types.png', dpi=150, bbox_inches='tight')
    print("\nGenerated 5 samples of each curve type")
    print("Visualization saved to 'curve_types.png'")


def demo_difficulty_distribution():
    """Demonstrate difficulty proxy distribution."""
    print("\n" + "=" * 60)
    print("DEMO 3: Difficulty Proxy Distribution")
    print("=" * 60)

    # Generate many curves
    generator = ParametricCurveGenerator()
    curves = generator.generate_dataset(num_curves=1000, length=100)

    # Compute difficulty proxies
    proxy_calc = DifficultyProxy()
    proxy_calc.fit_standardization(curves[:500])  # Fit on subset

    proxies = []
    for curve in curves:
        proxy = proxy_calc.compute(curve[:30], standardize=False)
        proxies.append(proxy)

    proxies = np.array(proxies)

    # Visualize distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    labels = ['Progress Rate', 'Non-linearity', 'Volatility']
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.hist(proxies[:, i], bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(label)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {label}')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean = proxies[:, i].mean()
        std = proxies[:, i].std()
        ax.axvline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig('difficulty_distribution.png', dpi=150, bbox_inches='tight')
    print("\nComputed difficulty proxies for 1000 curves")
    print("Visualization saved to 'difficulty_distribution.png'")

    print("\nDifficulty Proxy Statistics:")
    print(f"  Progress Rate:  μ={proxies[:, 0].mean():.4f}, σ={proxies[:, 0].std():.4f}")
    print(f"  Non-linearity:  μ={proxies[:, 1].mean():.4f}, σ={proxies[:, 1].std():.4f}")
    print(f"  Volatility:     μ={proxies[:, 2].mean():.4f}, σ={proxies[:, 2].std():.4f}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("DA-PFN Framework Demo")
    print("=" * 60)

    try:
        # Demo 1: Difficulty proxy
        demo_difficulty_proxy()

        # Demo 2: Curve generation
        demo_curve_generation()

        # Demo 3: Difficulty distribution
        demo_difficulty_distribution()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
