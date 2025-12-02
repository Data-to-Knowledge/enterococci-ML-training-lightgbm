"""
Apply probability threshold optimization to SAVED fold predictions.

This ensures we're optimizing thresholds on the EXACT SAME predictions
that were used in the production baseline evaluation.
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


def evaluate_with_threshold(y_true, y_prob, threshold, exceedance_threshold=280):
    """Evaluate using probability threshold."""
    y_true_binary = (y_true >= exceedance_threshold).astype(int)
    y_pred_binary = (y_prob >= threshold).astype(int)

    TP = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    FP = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
    TN = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
    FN = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    return {
        'TP': int(TP),
        'FP': int(FP),
        'TN': int(TN),
        'FN': int(FN),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'threshold': threshold
    }


def main():
    logger.info("="*80)
    logger.info("THRESHOLD OPTIMIZATION ON SAVED PREDICTIONS")
    logger.info("="*80)

    # Load saved fold predictions
    fold_preds = pd.read_csv("reports/evaluation_results_fold_predictions.csv")
    logger.info(f"\nLoaded {len(fold_preds)} predictions from evaluation_results_fold_predictions.csv")
    logger.info(f"Sites: {fold_preds['SITE_NAME'].nunique()}")

    # Check if prob_exceed_280 column exists
    if 'prob_exceed_280' not in fold_preds.columns:
        logger.error("ERROR: prob_exceed_280 column not found in fold predictions!")
        logger.error("Available columns: " + str(fold_preds.columns.tolist()))
        logger.error("\nYou need to re-run the pipeline to generate predictions with prob_exceed_280.")
        return

    # Extract data
    y_true = fold_preds['Enterococci'].values
    y_pred_point = fold_preds['predictions'].values
    y_prob = fold_preds['prob_exceed_280'].values

    # Baseline (point forecast > 280)
    logger.info("\n" + "="*80)
    logger.info("BASELINE: Point Forecast (predictions > 280)")
    logger.info("="*80)
    baseline = evaluate_with_threshold(y_true, y_pred_point, 280.0)
    logger.info(f"TP={baseline['TP']}, FP={baseline['FP']}, TN={baseline['TN']}, FN={baseline['FN']}")
    logger.info(f"Sensitivity={baseline['sensitivity']:.3f}, Specificity={baseline['specificity']:.3f}")

    # Optimize thresholds
    y_true_binary = (y_true >= 280).astype(int)

    logger.info("\n" + "="*80)
    logger.info("PROBABILITY THRESHOLD OPTIMIZATION")
    logger.info("="*80)

    results = {}

    # Try multiple cost ratios
    for cost_ratio in [5, 10, 20]:
        logger.info(f"\n{'─'*80}")
        logger.info(f"Cost Ratio = {cost_ratio}")
        logger.info(f"{'─'*80}")

        optimizer = ThresholdOptimizer(
            min_sensitivity=0.5,
            cost_ratio=cost_ratio,
            threshold_step=0.01,
            min_threshold=0.05
        )

        optimal = optimizer.optimize(y_true_binary, y_prob)

        # Evaluate each method
        for method_name in ['youden_j', 'cost_weighted', 'max_specificity']:
            threshold = optimal[method_name]
            metrics = evaluate_with_threshold(y_true, y_prob, threshold)

            if method_name not in results:
                results[method_name] = []
            results[method_name].append({
                'cost_ratio': cost_ratio,
                **metrics
            })

            logger.info(
                f"  {method_name:20s}: threshold={threshold:.3f}  "
                f"Sens={metrics['sensitivity']:.3f}  "
                f"Spec={metrics['specificity']:.3f}  "
                f"TP={metrics['TP']} FN={metrics['FN']}"
            )

    # Summary table
    logger.info("\n" + "="*80)
    logger.info("SUMMARY COMPARISON")
    logger.info("="*80)

    print("\nBASELINE (Point Forecast > 280):")
    print(f"  TP={baseline['TP']}, FP={baseline['FP']}, TN={baseline['TN']}, FN={baseline['FN']}")
    print(f"  Sensitivity={baseline['sensitivity']:.3f}, Specificity={baseline['specificity']:.3f}")

    print("\n" + "─"*80)
    for method_name in ['youden_j', 'cost_weighted', 'max_specificity']:
        print(f"\n{method_name.upper().replace('_', ' ')}:")
        for result in results[method_name]:
            print(
                f"  CR={result['cost_ratio']:2d}: "
                f"threshold={result['threshold']:.3f}  "
                f"Sens={result['sensitivity']:.3f}  "
                f"Spec={result['specificity']:.3f}  "
                f"TP={result['TP']:2d} FP={result['FP']:3d} FN={result['FN']:2d}"
            )

    # Save results
    output = {
        'baseline': baseline,
        'optimized_methods': results,
        'dataset_info': {
            'total_samples': len(fold_preds),
            'sites': int(fold_preds['SITE_NAME'].nunique()),
            'exceedances': int((y_true >= 280).sum())
        }
    }

    output_path = project_root / "reports" / "threshold_optimization" / "saved_predictions_threshold_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

    logger.info(f"\n\nResults saved to: {output_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
