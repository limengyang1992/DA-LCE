"""
Evaluation Metrics for Learning Curve Extrapolation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


class LCEvaluator:
    """Evaluator for learning curve extrapolation."""

    @staticmethod
    def compute_mape(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Mean Absolute Percentage Error.

        Args:
            predictions: Predicted values [N]
            targets: True values [N]

        Returns:
            MAPE score
        """
        # Avoid division by zero
        mask = targets != 0
        if mask.sum() == 0:
            return 0.0

        ape = np.abs((targets[mask] - predictions[mask]) / targets[mask])
        return np.mean(ape)

    @staticmethod
    def compute_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Mean Absolute Error.

        Args:
            predictions: Predicted values [N]
            targets: True values [N]

        Returns:
            MAE score
        """
        return np.mean(np.abs(predictions - targets))

    @staticmethod
    def compute_rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Root Mean Squared Error.

        Args:
            predictions: Predicted values [N]
            targets: True values [N]

        Returns:
            RMSE score
        """
        return np.sqrt(np.mean((predictions - targets) ** 2))

    @staticmethod
    def compute_nll(log_probs: np.ndarray) -> float:
        """
        Compute Negative Log-Likelihood.

        Args:
            log_probs: Log probabilities of true values [N]

        Returns:
            NLL score
        """
        return -np.mean(log_probs)

    @staticmethod
    def compute_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Spearman correlation coefficient.

        Args:
            predictions: Predicted values [N]
            targets: True values [N]

        Returns:
            Correlation coefficient
        """
        if len(predictions) < 2:
            return 0.0

        corr, _ = stats.spearmanr(predictions, targets)
        return corr if not np.isnan(corr) else 0.0

    def evaluate_predictions(self,
                            predictions: np.ndarray,
                            targets: np.ndarray,
                            log_probs: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            predictions: Predicted values [N]
            targets: True values [N]
            log_probs: Log probabilities (optional) [N]

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mape': self.compute_mape(predictions, targets),
            'mae': self.compute_mae(predictions, targets),
            'rmse': self.compute_rmse(predictions, targets),
            'correlation': self.compute_correlation(predictions, targets)
        }

        if log_probs is not None:
            metrics['nll'] = self.compute_nll(log_probs)

        return metrics

    @staticmethod
    def stratify_by_difficulty(curves: List[np.ndarray],
                               predictions: np.ndarray,
                               targets: np.ndarray,
                               difficulty_proxies: np.ndarray,
                               n_strata: int = 3) -> Dict[str, Dict]:
        """
        Stratify evaluation by task difficulty.

        Args:
            curves: List of learning curves
            predictions: Predicted values [N]
            targets: True values [N]
            difficulty_proxies: Difficulty proxy vectors [N, 3]
            n_strata: Number of difficulty strata

        Returns:
            Metrics for each stratum
        """
        from sklearn.cluster import KMeans

        # Cluster by difficulty (using all 3 components)
        kmeans = KMeans(n_clusters=n_strata, random_state=42)
        difficulty_labels = kmeans.fit_predict(difficulty_proxies)

        # Sort clusters by difficulty (using first component - progress)
        cluster_means = []
        for i in range(n_strata):
            mask = difficulty_labels == i
            if mask.sum() > 0:
                # Lower progress = harder task
                mean_progress = difficulty_proxies[mask, 0].mean()
                cluster_means.append((i, mean_progress))

        # Sort by progress (ascending = easy to hard)
        cluster_means.sort(key=lambda x: x[1], reverse=True)
        cluster_order = [x[0] for x in cluster_means]

        # Map to difficulty names
        difficulty_names = ['Easy', 'Medium', 'Hard']
        if n_strata == 2:
            difficulty_names = ['Easy', 'Hard']
        elif n_strata > 3:
            difficulty_names = [f'Stratum_{i+1}' for i in range(n_strata)]

        # Compute metrics for each stratum
        evaluator = LCEvaluator()
        results = {}

        for idx, cluster_id in enumerate(cluster_order[:len(difficulty_names)]):
            mask = difficulty_labels == cluster_id
            if mask.sum() == 0:
                continue

            stratum_name = difficulty_names[idx]
            stratum_metrics = evaluator.evaluate_predictions(
                predictions[mask],
                targets[mask]
            )
            stratum_metrics['count'] = mask.sum()

            results[stratum_name] = stratum_metrics

        return results


class EarlyStoppingSimulator:
    """
    Simulate early stopping for AutoML.
    Computes anytime regret curves.
    """

    def __init__(self, model, difficulty_proxy_calc, discretizer):
        """
        Args:
            model: Trained CD-PFN model
            difficulty_proxy_calc: Difficulty proxy calculator
            discretizer: Performance discretizer
        """
        self.model = model
        self.proxy_calc = difficulty_proxy_calc
        self.discretizer = discretizer

    def simulate_sequential_training(self,
                                     curves: List[np.ndarray],
                                     observation_budget: int = 20,
                                     confidence_threshold: float = 0.8) -> Tuple[List[int], List[float]]:
        """
        Simulate sequential model selection with early stopping.

        Args:
            curves: List of learning curves to evaluate
            observation_budget: Budget for early observations
            confidence_threshold: Threshold for stopping decision

        Returns:
            stopped_at: List of stopping epochs for each curve
            final_predictions: List of final predicted performances
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        stopped_at = []
        final_predictions = []

        with torch.no_grad():
            for curve in curves:
                # Observe up to budget
                observe_epochs = min(observation_budget, len(curve) - 1)

                # Prepare input
                epochs = torch.arange(observe_epochs, dtype=torch.float32).unsqueeze(0).to(device)
                performances = torch.from_numpy(curve[:observe_epochs]).unsqueeze(0).to(device)

                # Compute difficulty proxy
                difficulty_proxy = self.proxy_calc.compute(curve[:observe_epochs])
                difficulty_proxy = torch.from_numpy(difficulty_proxy).unsqueeze(0).to(device)

                # Predict
                bin_edges = self.discretizer.get_bin_edges().to(device)
                prediction = self.model.predict_mean(
                    epochs, performances, difficulty_proxy, bin_edges
                )

                stopped_at.append(observe_epochs)
                final_predictions.append(prediction.item())

        return stopped_at, final_predictions

    def compute_anytime_regret(self,
                              curves: List[np.ndarray],
                              max_budget: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anytime regret curve.

        Args:
            curves: List of learning curves
            max_budget: Maximum computational budget

        Returns:
            budgets: Array of budget points
            regrets: Corresponding regret values
        """
        # Ground truth: best final performance
        true_finals = np.array([curve[-1] for curve in curves])
        best_true = np.max(true_finals)

        budgets = []
        regrets = []

        # Simulate for different budgets
        for budget in range(10, max_budget + 1, 5):
            stopped_at, predictions = self.simulate_sequential_training(
                curves, observation_budget=budget
            )

            # Compute cumulative budget
            total_budget = sum(stopped_at)
            budgets.append(total_budget)

            # Predicted best
            predicted_best_idx = np.argmax(predictions)
            actual_performance = curves[predicted_best_idx][-1]

            # Regret
            regret = best_true - actual_performance
            regrets.append(regret)

        return np.array(budgets), np.array(regrets)
