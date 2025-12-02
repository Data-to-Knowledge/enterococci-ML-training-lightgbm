"""
Comprehensive comparison of classification approaches for beach water quality forecasting.

BASELINE:
- Median of RAW quantiles > 280 (best point forecast performance)

PROBABILITY-BASED APPROACHES (using monotonic quantiles for CDF):
1. Youden's J Index - Maximizes sensitivity + specificity
2. Cost-weighted - Balances cost of false negatives vs false positives
3. Max Specificity - High specificity while maintaining minimum sensitivity

All approaches use prob_exceed_280 calculated from monotonic quantiles via CDF.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.evaluation.threshold_optimization import ThresholdOptimizer
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def calculate_metrics(y_true, y_pred_binary, threshold_type=""):
    """Calculate confusion matrix and performance metrics."""
    y_true_binary = (y_true >= 280).astype(int)

    TP = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    FP = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
    TN = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
    FN = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    return {
        'TP': int(TP),
        'FP': int(FP),
        'TN': int(TN),
        'FN': int(FN),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'precision': precision,
        'threshold_type': threshold_type
    }


def print_metrics(metrics, label):
    """Print metrics in a formatted way."""
    print(f"\n{label}")
    print(f"  TP={metrics['TP']:3d}  FP={metrics['FP']:3d}  TN={metrics['TN']:3d}  FN={metrics['FN']:3d}")
    print(f"  Sensitivity: {metrics['sensitivity']:.3f}  ({metrics['TP']}/{metrics['TP']+metrics['FN']} exceedances caught)")
    print(f"  Specificity: {metrics['specificity']:.3f}  ({metrics['TN']}/{metrics['TN']+metrics['FP']} safe days correct)")
    print(f"  Accuracy:    {metrics['accuracy']:.3f}")
    print(f"  Precision:   {metrics['precision']:.3f}  ({metrics['TP']}/{metrics['TP']+metrics['FP']} warnings correct)")


def main():
    logger.info("="*80)
    logger.info("COMPREHENSIVE BASELINE vs PROBABILITY THRESHOLD COMPARISON")
    logger.info("="*80)

    # Load saved fold predictions
    fold_preds = pd.read_csv("reports/evaluation_results_fold_predictions.csv")
    logger.info(f"\nLoaded {len(fold_preds)} predictions from evaluation_results_fold_predictions.csv")
    logger.info(f"Sites: {fold_preds['SITE_NAME'].nunique()}")

    # Extract data
    y_true = fold_preds['Enterococci'].values
    y_true_binary = (y_true >= 280).astype(int)

    # Calculate median of RAW quantiles for baseline
    raw_quantile_cols = [col for col in fold_preds.columns if col.endswith('_raw')]
    median_raw = fold_preds[raw_quantile_cols].median(axis=1).values

    # Get probability from monotonic quantiles (for CDF calculation)
    y_prob = fold_preds['prob_exceed_280'].values

    logger.info(f"\nTotal samples: {len(fold_preds)}")
    logger.info(f"True exceedances (>=280): {y_true_binary.sum()}")
    logger.info(f"True safe (<280): {(~y_true_binary.astype(bool)).sum()}")

    # ========================================================================
    # BASELINE: Median of RAW quantiles > 280
    # ========================================================================
    print("\n" + "="*80)
    print("BASELINE: Point Forecast (Median of RAW quantiles > 280)")
    print("="*80)

    baseline_binary = (median_raw >= 280).astype(int)
    baseline_metrics = calculate_metrics(y_true, baseline_binary, "point_forecast_raw")
    print_metrics(baseline_metrics, "Performance:")

    # ========================================================================
    # PROBABILITY-BASED APPROACHES
    # ========================================================================
    print("\n" + "="*80)
    print("PROBABILITY-BASED APPROACHES")
    print("Using prob_exceed_280 from monotonic quantiles (CDF method)")
    print("="*80)

    results = {
        'baseline': baseline_metrics,
        'probability_methods': {}
    }

    # Configuration for different optimization methods
    configs = [
        {
            'name': 'Youden\'s J Index',
            'description': 'Maximizes (Sensitivity + Specificity - 1)',
            'method': 'youden_j',
            'cost_ratio': None,
            'min_sensitivity': 0.5
        },
        {
            'name': 'Cost-Weighted (CR=10)',
            'description': 'False negative costs 10x more than false positive',
            'method': 'cost_weighted',
            'cost_ratio': 10,
            'min_sensitivity': 0.5
        },
        {
            'name': 'Max Specificity (MinSens=0.65)',
            'description': 'Maximize specificity while maintaining 65% sensitivity',
            'method': 'max_specificity',
            'cost_ratio': 10,  # Still needed for optimizer
            'min_sensitivity': 0.65
        }
    ]

    for config in configs:
        print(f"\n{'-'*80}")
        print(f"{config['name']}")
        print(f"  {config['description']}")
        print(f"{'-'*80}")

        # Initialize optimizer
        optimizer = ThresholdOptimizer(
            min_sensitivity=config['min_sensitivity'],
            cost_ratio=config['cost_ratio'] if config['cost_ratio'] else 10,
            threshold_step=0.01,
            min_threshold=0.05
        )

        # Optimize thresholds
        optimal_thresholds = optimizer.optimize(y_true_binary, y_prob)
        optimal_threshold = optimal_thresholds[config['method']]

        # Apply threshold
        prob_binary = (y_prob >= optimal_threshold).astype(int)
        prob_metrics = calculate_metrics(y_true, prob_binary, config['method'])
        prob_metrics['probability_threshold'] = optimal_threshold

        # Store results
        results['probability_methods'][config['method']] = prob_metrics

        # Print results
        print(f"\nOptimal probability threshold: {optimal_threshold:.3f}")
        print_metrics(prob_metrics, "Performance:")

        # Compare to baseline
        print(f"\nComparison to baseline:")
        print(f"  Change TP: {prob_metrics['TP'] - baseline_metrics['TP']:+3d}")
        print(f"  Change FP: {prob_metrics['FP'] - baseline_metrics['FP']:+3d}")
        print(f"  Change FN: {prob_metrics['FN'] - baseline_metrics['FN']:+3d}")
        print(f"  Change Sensitivity: {prob_metrics['sensitivity'] - baseline_metrics['sensitivity']:+.3f}")
        print(f"  Change Specificity: {prob_metrics['specificity'] - baseline_metrics['specificity']:+.3f}")
        print(f"  Change Accuracy:    {prob_metrics['accuracy'] - baseline_metrics['accuracy']:+.3f}")

    # ========================================================================
    # SUMMARY COMPARISON TABLE
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)

    print("\n{:<30s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>8s}".format(
        "Method", "TP", "FP", "TN", "FN", "Sens", "Spec", "Thresh"))
    print("-"*80)

    # Baseline
    print("{:<30s} {:>6d} {:>6d} {:>6d} {:>6d} {:>6.3f} {:>6.3f} {:>8s}".format(
        "BASELINE (raw median>280)",
        baseline_metrics['TP'],
        baseline_metrics['FP'],
        baseline_metrics['TN'],
        baseline_metrics['FN'],
        baseline_metrics['sensitivity'],
        baseline_metrics['specificity'],
        "280"
    ))

    # Probability methods
    for config in configs:
        method = config['method']
        metrics = results['probability_methods'][method]
        print("{:<30s} {:>6d} {:>6d} {:>6d} {:>6d} {:>6.3f} {:>6.3f} {:>8.3f}".format(
            config['name'][:30],
            metrics['TP'],
            metrics['FP'],
            metrics['TN'],
            metrics['FN'],
            metrics['sensitivity'],
            metrics['specificity'],
            metrics['probability_threshold']
        ))

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Find best method by different criteria
    all_methods = [baseline_metrics] + list(results['probability_methods'].values())

    best_sensitivity = max(all_methods, key=lambda x: x['sensitivity'])
    best_specificity = max(all_methods, key=lambda x: x['specificity'])
    best_accuracy = max(all_methods, key=lambda x: x['accuracy'])

    print(f"\nBest Sensitivity: {best_sensitivity['threshold_type']} ({best_sensitivity['sensitivity']:.3f})")
    print(f"Best Specificity: {best_specificity['threshold_type']} ({best_specificity['specificity']:.3f})")
    print(f"Best Accuracy:    {best_accuracy['threshold_type']} ({best_accuracy['accuracy']:.3f})")

    print("\nFor public health beach warnings:")
    print("  - Prioritize SENSITIVITY if goal is to catch all unsafe days (minimize missed exceedances)")
    print("  - Prioritize SPECIFICITY if goal is to minimize false alarms (maintain public trust)")
    print("  - Use COST-WEIGHTED if you can quantify the relative cost of false negatives vs false positives")
    print("  - Use YOUDEN'S J for balanced performance")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    output_path = project_root / "reports" / "threshold_optimization" / "comprehensive_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare JSON-serializable results
    json_results = {
        'baseline': {
            'method': 'Median of RAW quantiles > 280',
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in baseline_metrics.items()}
        },
        'probability_methods': {
            config['name']: {
                'description': config['description'],
                'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in results['probability_methods'][config['method']].items()}
            }
            for config in configs
        },
        'dataset_info': {
            'total_samples': int(len(fold_preds)),
            'sites': int(fold_preds['SITE_NAME'].nunique()),
            'exceedances': int(y_true_binary.sum()),
            'safe_days': int((~y_true_binary.astype(bool)).sum())
        }
    }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=4)

    logger.info(f"\n\nResults saved to: {output_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
