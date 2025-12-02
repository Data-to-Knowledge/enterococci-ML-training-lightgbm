"""
Test script for probability threshold optimization in cross-validation.

This script:
1. Loads the trained probabilistic model
2. Runs time-series cross-validation
3. Uses prob_exceed_280 instead of point forecasts for classification
4. Optimizes probability threshold using 3 methods:
   - Youden's J (balanced)
   - Cost-weighted (public health focused)
   - Max specificity (minimize false alarms)
5. Reports per-fold and overall metrics
6. Generates visualization plots

Usage:
    python scripts/test_probability_threshold_optimization.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config.paths import TRAINING_DATA_PATH
from src.evaluation.cross_validation import TimeSeriesCV
from src.evaluation.threshold_optimization import ThresholdOptimizer
from src.data.preprocessing import Preprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.probabilistic.main_probabilistic import ProbabilisticForecastingModel
from src.utils.logging import setup_logger
import yaml

logger = setup_logger(__name__)


def load_config(config_path: Path) -> dict:
    """Load main configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_with_probability_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    exceedance_threshold: float = 280
) -> dict:
    """
    Evaluate using probability threshold instead of point forecast.

    Parameters
    ----------
    y_true : np.ndarray
        True enterococci values.
    y_prob : np.ndarray
        Predicted probabilities of exceedance.
    threshold : float
        Probability threshold for classification.
    exceedance_threshold : float
        Enterococci threshold for exceedance (default 280).

    Returns
    -------
    dict
        Confusion matrix and metrics.
    """
    # Convert to binary labels
    y_true_binary = (y_true >= exceedance_threshold).astype(int)
    y_pred_binary = (y_prob >= threshold).astype(int)

    # Confusion matrix
    TP = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    FP = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
    TN = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
    FN = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

    # Metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    youden_j = sensitivity + specificity - 1

    return {
        'TP': int(TP),
        'FP': int(FP),
        'TN': int(TN),
        'FN': int(FN),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'youden_j': youden_j,
        'threshold_used': threshold
    }


def main():
    logger.info("="*80)
    logger.info("PROBABILITY THRESHOLD OPTIMIZATION - CROSS-VALIDATION TEST")
    logger.info("="*80)

    # Load configuration
    config_path = project_root / "src" / "config" / "main_config.yaml"
    config = load_config(config_path)

    # Initialize components
    preprocessor = Preprocessor()
    feature_engineer = FeatureEngineer()

    # Load and prepare data
    logger.info("\n" + "="*80)
    logger.info("LOADING AND PREPARING DATA")
    logger.info("="*80)

    data = pd.read_csv(TRAINING_DATA_PATH)
    logger.info(f"Raw data shape: {data.shape}")

    data = preprocessor.clean_data(data)
    data = feature_engineer.engineer_features(data)
    data = preprocessor.transform_catergorical_variable_type(data)
    data = preprocessor.label_encode(data)
    data = preprocessor.fill_missing_values(data)
    data['Enterococci'] = preprocessor.set_max_target_value(data['Enterococci'])

    logger.info(f"Processed data shape: {data.shape}")

    # Initialize time-series CV
    cv_config = config['evaluation']['cross_validation']
    cv = TimeSeriesCV(cv_config)

    # Get CV splits
    splits = cv.split(data, date_column='DateTime')
    logger.info(f"Generated {len(splits)} CV folds")

    # Results storage
    fold_results = {}
    all_y_true = []
    all_y_prob = []
    all_y_pred_point = []
    all_site_names = []

    # ========================================================================
    # CROSS-VALIDATION LOOP
    # ========================================================================
    for fold_idx, (train_data, test_data) in enumerate(splits, 1):
        logger.info("\n" + "="*80)
        logger.info(f"FOLD {fold_idx}/{len(splits)}")
        logger.info("="*80)

        # Prepare data
        X_train = train_data.drop('Enterococci', axis=1)
        y_train = train_data['Enterococci']
        X_test = test_data.drop('Enterococci', axis=1)
        y_test = test_data['Enterococci']

        # Store site names for per-site optimization
        if 'SITE_NAME' in test_data.columns:
            site_names = test_data['SITE_NAME'].values
        else:
            site_names = np.array(['Unknown'] * len(y_test))

        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        logger.info(f"Test exceedances: {(y_test >= 280).sum()} / {len(y_test)} ({100*(y_test>=280).mean():.1f}%)")

        # Train model
        logger.info("\nTraining probabilistic model...")
        model = ProbabilisticForecastingModel(config)
        model.train(pd.concat([X_train, y_train], axis=1))

        # Get predictions
        logger.info("Generating predictions...")
        predictions = model.predict(X_test)

        # Extract relevant columns
        y_prob = predictions['prob_exceed_280'].values
        y_pred_point = predictions['predictions'].values

        # Collect for overall optimization
        all_y_true.extend(y_test.values)
        all_y_prob.extend(y_prob)
        all_y_pred_point.extend(y_pred_point)
        all_site_names.extend(site_names)

        # ──────────────────────────────────────────────────────────────────
        # THRESHOLD OPTIMIZATION FOR THIS FOLD
        # ──────────────────────────────────────────────────────────────────
        logger.info("\n" + "-"*80)
        logger.info(f"OPTIMIZING THRESHOLDS FOR FOLD {fold_idx}")
        logger.info("-"*80)

        # Convert to binary labels
        y_true_binary = (y_test >= 280).values.astype(int)

        # Initialize optimizer
        optimizer = ThresholdOptimizer(
            min_sensitivity=0.5,
            cost_ratio=7.0,
            threshold_step=0.01
        )

        # Find optimal thresholds
        optimal_thresholds = optimizer.optimize(y_true_binary, y_prob)

        # Evaluate with each method
        fold_metrics = {}
        for method_name in ['youden_j', 'cost_weighted', 'max_specificity']:
            threshold = optimal_thresholds[method_name]
            metrics = evaluate_with_probability_threshold(
                y_test.values, y_prob, threshold
            )
            fold_metrics[method_name] = metrics

            logger.info(
                f"{method_name:20s}: threshold={threshold:.3f}  "
                f"Sens={metrics['sensitivity']:.3f}  "
                f"Spec={metrics['specificity']:.3f}  "
                f"Youden={metrics['youden_j']:.3f}"
            )

        # Also evaluate with point forecast (baseline)
        y_pred_binary_point = (y_pred_point >= 280).astype(int)
        TP_point = np.sum((y_pred_binary_point == 1) & (y_true_binary == 1))
        FP_point = np.sum((y_pred_binary_point == 1) & (y_true_binary == 0))
        TN_point = np.sum((y_pred_binary_point == 0) & (y_true_binary == 0))
        FN_point = np.sum((y_pred_binary_point == 0) & (y_true_binary == 1))

        fold_metrics['point_forecast_baseline'] = {
            'TP': int(TP_point),
            'FP': int(FP_point),
            'TN': int(TN_point),
            'FN': int(FN_point),
            'sensitivity': TP_point / (TP_point + FN_point) if (TP_point + FN_point) > 0 else 0.0,
            'specificity': TN_point / (TN_point + FP_point) if (TN_point + FP_point) > 0 else 0.0,
            'threshold_used': 280.0
        }

        logger.info(
            f"{'point_forecast_baseline':20s}: threshold=280 (direct)  "
            f"Sens={fold_metrics['point_forecast_baseline']['sensitivity']:.3f}  "
            f"Spec={fold_metrics['point_forecast_baseline']['specificity']:.3f}"
        )

        # Save fold results
        fold_results[f'fold_{fold_idx}'] = {
            'optimal_thresholds': {k: v for k, v in optimal_thresholds.items() if k != 'metrics'},
            'metrics': fold_metrics
        }

        # Generate plots for this fold
        plot_path = project_root / "reports" / "threshold_optimization" / f"fold_{fold_idx}_threshold_curves.png"
        optimizer.plot_threshold_curves(y_true_binary, y_prob, output_path=plot_path)

    # ========================================================================
    # OVERALL OPTIMIZATION (ACROSS ALL FOLDS)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("OVERALL THRESHOLD OPTIMIZATION (ALL FOLDS COMBINED)")
    logger.info("="*80)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred_point = np.array(all_y_pred_point)
    all_site_names = np.array(all_site_names)
    all_y_true_binary = (all_y_true >= 280).astype(int)

    logger.info(f"Total samples: {len(all_y_true)}")
    logger.info(f"Total exceedances: {all_y_true_binary.sum()} ({100*all_y_true_binary.mean():.2f}%)")

    # Optimize overall
    overall_optimizer = ThresholdOptimizer(min_sensitivity=0.5, cost_ratio=7.0)
    overall_optimal = overall_optimizer.optimize(all_y_true_binary, all_y_prob)

    # Evaluate overall with each method
    overall_metrics = {}
    for method_name in ['youden_j', 'cost_weighted', 'max_specificity']:
        threshold = overall_optimal[method_name]
        metrics = evaluate_with_probability_threshold(
            all_y_true, all_y_prob, threshold
        )
        overall_metrics[method_name] = metrics

    # Baseline (point forecast)
    y_pred_binary_point = (all_y_pred_point >= 280).astype(int)
    TP_point = np.sum((y_pred_binary_point == 1) & (all_y_true_binary == 1))
    FP_point = np.sum((y_pred_binary_point == 1) & (all_y_true_binary == 0))
    TN_point = np.sum((y_pred_binary_point == 0) & (all_y_true_binary == 0))
    FN_point = np.sum((y_pred_binary_point == 0) & (all_y_true_binary == 1))

    overall_metrics['point_forecast_baseline'] = {
        'TP': int(TP_point),
        'FP': int(FP_point),
        'TN': int(TN_point),
        'FN': int(FN_point),
        'sensitivity': TP_point / (TP_point + FN_point) if (TP_point + FN_point) > 0 else 0.0,
        'specificity': TN_point / (TN_point + FP_point) if (TN_point + FP_point) > 0 else 0.0,
        'threshold_used': 280.0
    }

    # Plot overall
    overall_plot_path = project_root / "reports" / "threshold_optimization" / "overall_threshold_curves.png"
    overall_optimizer.plot_threshold_curves(all_y_true_binary, all_y_prob, output_path=overall_plot_path)

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    results_dict = {
        'fold_results': fold_results,
        'overall': {
            'optimal_thresholds': {k: v for k, v in overall_optimal.items() if k != 'metrics'},
            'metrics': overall_metrics
        },
        'config': {
            'min_sensitivity': 0.5,
            'cost_ratio': 7.0,
            'threshold_step': 0.01,
            'exceedance_threshold': 280
        }
    }

    output_path = project_root / "reports" / "threshold_optimization" / "probability_threshold_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=4)

    logger.info(f"\nResults saved to: {output_path}")

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("FINAL COMPARISON: POINT FORECAST vs PROBABILITY THRESHOLD METHODS")
    logger.info("="*80)

    summary_data = []
    for method in ['point_forecast_baseline', 'youden_j', 'cost_weighted', 'max_specificity']:
        m = overall_metrics[method]
        summary_data.append({
            'Method': method,
            'Threshold': f"{m['threshold_used']:.3f}" if 'threshold_used' in m else "280 (point)",
            'Sensitivity': f"{m['sensitivity']:.3f}",
            'Specificity': f"{m['specificity']:.3f}",
            'TP': m['TP'],
            'FP': m['FP'],
            'TN': m['TN'],
            'FN': m['FN']
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    logger.info("\n" + "="*80)
    logger.info("THRESHOLD OPTIMIZATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
