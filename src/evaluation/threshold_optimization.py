"""
Probability Threshold Optimization for Exceedance Classification

This module implements scientifically rigorous methods to determine the optimal
probability threshold for classifying water quality exceedances (Enterococci > 280 MPN/100mL).

Key Approaches:
1. Constraint-based search: Test thresholds from 0.01 to 0.99 with sensitivity ≥ 0.5
2. Optimization criteria:
   - Option A: Youden's J statistic (balanced sensitivity + specificity)
   - Option B: Cost-weighted (public health focus, FN more costly than FP)
   - Option C: Maximize specificity (minimize false alarms while meeting safety)

Author: Water Quality Forecasting Team
Date: 2025-12-02
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class ThresholdOptimizer:
    """
    Optimize probability threshold for exceedance classification.

    Given predicted probabilities and true labels, finds the optimal threshold
    that maximizes performance while meeting regulatory constraints.
    """

    def __init__(
        self,
        min_sensitivity: float = 0.5,
        cost_ratio: float = 7.0,
        threshold_step: float = 0.01,
        min_threshold: float = 0.0
    ):
        """
        Initialize threshold optimizer.

        Parameters
        ----------
        min_sensitivity : float
            Minimum required sensitivity (recall) for regulatory compliance.
            Default: 0.5 (50% of exceedances must be caught).
        cost_ratio : float
            Ratio of False Negative cost to False Positive cost.
            Higher values = missing exceedances is more costly than false alarms.
            Default: 7.0 (FN is 7x more costly than FP for public health).
        threshold_step : float
            Step size for threshold search grid.
            Default: 0.01 (test thresholds from 0.01 to 0.99 in 0.01 increments).
        """
        self.min_sensitivity = min_sensitivity
        self.cost_ratio = cost_ratio
        self.threshold_step = threshold_step
        self.min_threshold = min_threshold
        self.results = None

        logger.info(
            f"ThresholdOptimizer initialized: "
            f"min_sensitivity={min_sensitivity}, "
            f"cost_ratio={cost_ratio}, "
            f"step={threshold_step}, "
            f"min_threshold={min_threshold}"
        )

    def optimize(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        site_names: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Find optimal thresholds using multiple criteria.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels (1 = exceedance, 0 = safe).
        y_prob : np.ndarray
            Predicted probabilities of exceedance P(Y > 280).
        site_names : np.ndarray, optional
            Site names for per-site threshold optimization.

        Returns
        -------
        dict
            Dictionary with optimal thresholds for each method:
            - 'youden_j': Threshold maximizing Youden's J statistic
            - 'cost_weighted': Threshold maximizing cost-weighted score
            - 'max_specificity': Threshold maximizing specificity (given sensitivity ≥ min)
            - 'metrics': DataFrame with all threshold performances
        """
        logger.info(f"Optimizing thresholds for {len(y_true)} samples...")
        logger.info(f"  Exceedances: {y_true.sum()} ({100*y_true.mean():.2f}%)")
        logger.info(f"  Safe: {(1-y_true).sum()} ({100*(1-y_true.mean()):.2f}%)")

        # Step 1: Constraint-based search
        thresholds = np.arange(max(self.min_threshold, 0.01), 1.00, self.threshold_step)
        results = []

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)

            # Confusion matrix
            TP = np.sum((y_pred == 1) & (y_true == 1))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            TN = np.sum((y_pred == 0) & (y_true == 0))
            FN = np.sum((y_pred == 0) & (y_true == 1))

            # Metrics
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

            # Youden's J statistic
            youden_j = sensitivity + specificity - 1

            # Cost-weighted score
            cost_weighted = sensitivity - (1 / self.cost_ratio) * (1 - specificity)

            # F1 and F2 scores
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
            beta = 2.0
            f2 = (1 + beta**2) * precision * sensitivity / (beta**2 * precision + sensitivity) if (beta**2 * precision + sensitivity) > 0 else 0.0

            results.append({
                'threshold': thresh,
                'TP': TP,
                'FP': FP,
                'TN': TN,
                'FN': FN,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'youden_j': youden_j,
                'cost_weighted': cost_weighted,
                'f1_score': f1,
                'f2_score': f2,
                'meets_constraint': sensitivity >= self.min_sensitivity
            })

        self.results = pd.DataFrame(results)

        # Step 2: Find optimal thresholds
        valid_results = self.results[self.results['meets_constraint']]

        if len(valid_results) == 0:
            logger.warning(
                f"No thresholds meet min_sensitivity={self.min_sensitivity} constraint! "
                f"Using all thresholds instead."
            )
            valid_results = self.results

        optimal_thresholds = {
            'youden_j': valid_results.loc[valid_results['youden_j'].idxmax(), 'threshold'],
            'cost_weighted': valid_results.loc[valid_results['cost_weighted'].idxmax(), 'threshold'],
            'max_specificity': valid_results.loc[valid_results['specificity'].idxmax(), 'threshold'],
            'f1_score': valid_results.loc[valid_results['f1_score'].idxmax(), 'threshold'],
            'f2_score': valid_results.loc[valid_results['f2_score'].idxmax(), 'threshold'],
            'metrics': self.results
        }

        # Log optimal thresholds
        logger.info("\n" + "="*70)
        logger.info("OPTIMAL THRESHOLDS:")
        logger.info("="*70)
        for method, thresh in optimal_thresholds.items():
            if method == 'metrics':
                continue
            row = self.results[self.results['threshold'] == thresh].iloc[0]
            logger.info(
                f"{method:20s}: {thresh:.3f}  "
                f"(Sens={row['sensitivity']:.3f}, Spec={row['specificity']:.3f}, "
                f"Youden={row['youden_j']:.3f})"
            )
        logger.info("="*70 + "\n")

        return optimal_thresholds

    def optimize_per_site(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        site_names: np.ndarray,
        method: str = 'youden_j'
    ) -> Dict[str, float]:
        """
        Find optimal threshold for each site independently.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels.
        y_prob : np.ndarray
            Predicted probabilities.
        site_names : np.ndarray
            Site names for each sample.
        method : str
            Optimization method: 'youden_j', 'cost_weighted', or 'max_specificity'.

        Returns
        -------
        dict
            Site name -> optimal threshold mapping.
        """
        logger.info(f"Optimizing per-site thresholds using method: {method}")

        site_thresholds = {}
        unique_sites = np.unique(site_names)

        for site in unique_sites:
            mask = site_names == site
            site_y_true = y_true[mask]
            site_y_prob = y_prob[mask]

            if len(site_y_true) < 10:
                logger.warning(f"Site {site} has only {len(site_y_true)} samples - skipping")
                continue

            logger.info(f"\n  Site: {site} ({len(site_y_true)} samples)")
            optimal = self.optimize(site_y_true, site_y_prob)
            site_thresholds[site] = optimal[method]

        return site_thresholds

    def plot_threshold_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        output_path: Optional[Path] = None
    ) -> None:
        """
        Plot ROC curve, Precision-Recall curve, and threshold performance curves.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels.
        y_prob : np.ndarray
            Predicted probabilities.
        output_path : Path, optional
            Path to save the plot. If None, displays interactively.
        """
        if self.results is None:
            raise ValueError("Must run optimize() before plotting.")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[0, 0].axhline(y=self.min_sensitivity, color='r', linestyle='--',
                           label=f'Min Sensitivity = {self.min_sensitivity}')
        axes[0, 0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        axes[0, 0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        axes[0, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

        axes[0, 1].plot(recall, precision, 'g-', linewidth=2, label='PR Curve')
        axes[0, 1].axvline(x=self.min_sensitivity, color='r', linestyle='--',
                           label=f'Min Sensitivity = {self.min_sensitivity}')
        axes[0, 1].set_xlabel('Recall (Sensitivity)', fontsize=12)
        axes[0, 1].set_ylabel('Precision', fontsize=12)
        axes[0, 1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[0, 1].legend(loc='best')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Sensitivity & Specificity vs Threshold
        valid = self.results[self.results['meets_constraint']]

        axes[1, 0].plot(self.results['threshold'], self.results['sensitivity'],
                        'b-', linewidth=2, label='Sensitivity')
        axes[1, 0].plot(self.results['threshold'], self.results['specificity'],
                        'g-', linewidth=2, label='Specificity')
        axes[1, 0].axhline(y=self.min_sensitivity, color='r', linestyle='--',
                           label=f'Min Sensitivity = {self.min_sensitivity}')

        # Mark optimal thresholds
        if len(valid) > 0:
            opt_youden = valid.loc[valid['youden_j'].idxmax()]
            axes[1, 0].axvline(x=opt_youden['threshold'], color='purple',
                              linestyle=':', linewidth=2, label=f"Youden's J = {opt_youden['threshold']:.3f}")

        axes[1, 0].set_xlabel('Probability Threshold', fontsize=12)
        axes[1, 0].set_ylabel('Metric Value', fontsize=12)
        axes[1, 0].set_title('Sensitivity & Specificity vs Threshold', fontsize=14, fontweight='bold')
        axes[1, 0].legend(loc='best')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)

        # 4. Optimization Scores vs Threshold
        axes[1, 1].plot(self.results['threshold'], self.results['youden_j'],
                        'purple', linewidth=2, label="Youden's J")
        axes[1, 1].plot(self.results['threshold'], self.results['cost_weighted'],
                        'orange', linewidth=2, label=f'Cost-Weighted (ratio={self.cost_ratio})')
        axes[1, 1].plot(self.results['threshold'], self.results['f2_score'],
                        'brown', linewidth=2, label='F2 Score')

        axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel('Probability Threshold', fontsize=12)
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].set_title('Optimization Scores vs Threshold', fontsize=14, fontweight='bold')
        axes[1, 1].legend(loc='best')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, 1)

        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold optimization plot saved to {output_path}")
        else:
            plt.show()

        plt.close()

    def get_performance_at_threshold(
        self,
        threshold: float
    ) -> Dict[str, float]:
        """
        Get performance metrics at a specific threshold.

        Parameters
        ----------
        threshold : float
            Probability threshold to evaluate.

        Returns
        -------
        dict
            Performance metrics at this threshold.
        """
        if self.results is None:
            raise ValueError("Must run optimize() first.")

        closest_idx = (self.results['threshold'] - threshold).abs().idxmin()
        row = self.results.loc[closest_idx]

        return {
            'threshold': row['threshold'],
            'TP': int(row['TP']),
            'FP': int(row['FP']),
            'TN': int(row['TN']),
            'FN': int(row['FN']),
            'sensitivity': row['sensitivity'],
            'specificity': row['specificity'],
            'precision': row['precision'],
            'youden_j': row['youden_j'],
            'cost_weighted': row['cost_weighted'],
            'f1_score': row['f1_score'],
            'f2_score': row['f2_score'],
            'meets_constraint': row['meets_constraint']
        }
