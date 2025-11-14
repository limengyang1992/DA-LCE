"""
Example script for evaluating CD-PFN model.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.difficulty_proxy import DifficultyProxy
from src.cd_pfn import CDPFN, PerformanceDiscretizer
from src.evaluator import LCEvaluator
from src.data_generation import ParametricCurveGenerator
from src.utils import set_seed, get_device


def evaluate_model(model, test_curves, proxy_calc, discretizer, device):
    """Evaluate model on test curves."""
    model.eval()

    all_predictions = []
    all_targets = []
    all_proxies = []

    bin_edges = discretizer.get_bin_edges().to(device)

    with torch.no_grad():
        for curve in test_curves:
            # Use first 30 epochs as history
            history_length = 30
            target_epoch = len(curve) - 1

            # Prepare input
            epochs = torch.arange(history_length, dtype=torch.float32).unsqueeze(0).to(device)
            performances = torch.from_numpy(curve[:history_length]).unsqueeze(0).to(device)

            # Compute difficulty proxy
            difficulty_proxy = proxy_calc.compute(curve[:history_length])
            difficulty_proxy_tensor = torch.from_numpy(difficulty_proxy).unsqueeze(0).to(device)

            # Predict
            prediction = model.predict_mean(
                epochs, performances, difficulty_proxy_tensor, bin_edges
            )

            all_predictions.append(prediction.item())
            all_targets.append(curve[target_epoch])
            all_proxies.append(difficulty_proxy)

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_proxies = np.array(all_proxies)

    return all_predictions, all_targets, all_proxies


def main():
    set_seed(42)

    device = get_device()
    print(f"Using device: {device}")

    # Generate test data
    print("Generating test data...")
    generator = ParametricCurveGenerator()
    test_curves = generator.generate_dataset(
        num_curves=1000,
        length=100,
        noise_level=0.01
    )

    # Load difficulty proxy calculator
    print("Loading difficulty proxy calculator...")
    proxy_calc = DifficultyProxy()
    proxy_calc.load_statistics('difficulty_stats.npz')

    # Initialize model
    print("Loading model...")
    model = CDPFN(
        d_model=256,
        nhead=8,
        num_layers=6,
        num_bins=100,
        max_epochs=100
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Initialize discretizer
    discretizer = PerformanceDiscretizer(num_bins=100, min_val=0.0, max_val=1.0)

    # Evaluate
    print("Evaluating model...")
    predictions, targets, proxies = evaluate_model(
        model, test_curves, proxy_calc, discretizer, device
    )

    # Compute overall metrics
    evaluator = LCEvaluator()
    overall_metrics = evaluator.evaluate_predictions(predictions, targets)

    print("\n=== Overall Metrics ===")
    for metric, value in overall_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    # Stratify by difficulty
    print("\n=== Metrics by Difficulty ===")
    stratified_metrics = evaluator.stratify_by_difficulty(
        test_curves, predictions, targets, proxies, n_strata=3
    )

    for stratum, metrics in stratified_metrics.items():
        print(f"\n{stratum}:")
        for metric, value in metrics.items():
            if metric != 'count':
                print(f"  {metric.upper()}: {value:.4f}")
            else:
                print(f"  Count: {value}")

    # Save results
    results = {
        'overall': overall_metrics,
        'stratified': stratified_metrics
    }

    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print("\nResults saved to evaluation_results.json")


if __name__ == '__main__':
    main()
