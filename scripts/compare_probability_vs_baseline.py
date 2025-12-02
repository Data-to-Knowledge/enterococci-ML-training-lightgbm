"""
Comprehensive comparison: Probability threshold methods vs Point forecast baseline

This script:
1. Tests multiple cost ratios (5, 10, 20) to find scientifically justified thresholds
2. Enforces minimum threshold of 5% to avoid excessive false positives
3. Checks model calibration
4. Compares all approaches against baseline metrics
5. Generates site-level performance comparison

Usage:
    python scripts/compare_probability_vs_baseline.py
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


def check_calibration(y_true_binary, y_prob, n_bins=10):
    """
    Check if predicted probabilities are well-calibrated.

    Returns calibration data for plotting.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    calibration_data = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() > 0:
            observed_freq = y_true_binary[mask].mean()
            predicted_freq = y_prob[mask].mean()
            count = mask.sum()
            calibration_data.append({
                'bin': i,
                'bin_center': bin_centers[i],
                'predicted_prob': predicted_freq,
                'observed_freq': observed_freq,
                'count': count
            })

    return pd.DataFrame(calibration_data)


def evaluate_with_prob_threshold(y_true, y_prob, threshold, site_names=None):
    """Evaluate using probability threshold."""
    y_true_binary = (y_true >= 280).astype(int)
    y_pred_binary = (y_prob >= threshold).astype(int)

    # Overall metrics
    TP = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    FP = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
    TN = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
    FN = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    results = {
        'overall': {
            'TP': int(TP),
            'FP': int(FP),
            'TN': int(TN),
            'FN': int(FN),
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    }

    # Per-site metrics
    if site_names is not None:
        site_metrics = {}
        for site in np.unique(site_names):
            mask = site_names == site
            y_t_site = y_true_binary[mask]
            y_p_site = y_pred_binary[mask]

            TP_site = np.sum((y_p_site == 1) & (y_t_site == 1))
            FP_site = np.sum((y_p_site == 1) & (y_t_site == 0))
            TN_site = np.sum((y_p_site == 0) & (y_t_site == 0))
            FN_site = np.sum((y_p_site == 0) & (y_t_site == 1))

            sens_site = TP_site / (TP_site + FN_site) if (TP_site + FN_site) > 0 else 0.0
            spec_site = TN_site / (TN_site + FP_site) if (TN_site + FP_site) > 0 else 0.0

            site_metrics[site] = {
                'TP': int(TP_site),
                'FP': int(FP_site),
                'TN': int(TN_site),
                'FN': int(FN_site),
                'sensitivity': sens_site,
                'specificity': spec_site
            }

        results['site_metrics'] = site_metrics

    return results


def main():
    logger.info("="*80)
    logger.info("REFINED PROBABILITY THRESHOLD VS BASELINE COMPARISON")
    logger.info("="*80)

    # Load configuration
    config_path = project_root / "src" / "config" / "main_config.yaml"
    config = load_config(config_path)

    # Initialize components
    preprocessor = Preprocessor()
    feature_engineer = FeatureEngineer()

    # Load and prepare data
    logger.info("\nLoading and preparing data...")
    data = pd.read_csv(TRAINING_DATA_PATH)
    data = preprocessor.clean_data(data)
    data = feature_engineer.engineer_features(data)
    data = preprocessor.transform_catergorical_variable_type(data)
    # NOTE: Do NOT label_encode() before cv.split() - it needs text site names for filtering
    data = preprocessor.fill_missing_values(data)
    data['Enterococci'] = preprocessor.set_max_target_value(data['Enterococci'])

    # Initialize CV and get splits BEFORE encoding
    cv_config = config['evaluation']['cross_validation']
    cv = TimeSeriesCV(cv_config)
    splits_text = cv.split(data, date_column='DateTime')

    # NOW label encode the full dataset
    data = preprocessor.label_encode(data)

    # Re-create splits by matching indices from text splits
    splits = []
    for train_text, test_text in splits_text:
        train_encoded = data.loc[train_text.index]
        test_encoded = data.loc[test_text.index]
        splits.append((train_encoded, test_encoded))

    # Collect all predictions
    all_y_true = []
    all_y_prob = []
    all_y_pred_point = []
    all_site_names = []

    logger.info(f"\nRunning {len(splits)}-fold cross-validation...")

    for fold_idx, (train_data, test_data) in enumerate(splits, 1):
        logger.info(f"\n  Fold {fold_idx}/{len(splits)}: Training model...")

        X_train = train_data.drop('Enterococci', axis=1)
        y_train = train_data['Enterococci']
        X_test = test_data.drop('Enterococci', axis=1)
        y_test = test_data['Enterococci']

        # Get site names
        if 'SITE_NAME' in test_data.columns:
            site_names = test_data['SITE_NAME'].values
        else:
            site_names = np.array(['Unknown'] * len(y_test))

        # Train and predict
        model = ProbabilisticForecastingModel(config)
        model.train(pd.concat([X_train, y_train], axis=1))
        predictions = model.predict(X_test)

        y_prob = predictions['prob_exceed_280'].values
        y_pred_point = predictions['predictions'].values

        all_y_true.extend(y_test.values)
        all_y_prob.extend(y_prob)
        all_y_pred_point.extend(y_pred_point)
        all_site_names.extend(site_names)

    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred_point = np.array(all_y_pred_point)
    all_site_names = np.array(all_site_names)
    all_y_true_binary = (all_y_true >= 280).astype(int)

    logger.info(f"\nTotal test samples: {len(all_y_true)}")
    logger.info(f"Total exceedances: {all_y_true_binary.sum()} ({100*all_y_true_binary.mean():.2f}%)")

    # ========================================================================
    # CALIBRATION CHECK
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("MODEL CALIBRATION ANALYSIS")
    logger.info("="*80)

    calib_df = check_calibration(all_y_true_binary, all_y_prob, n_bins=10)
    logger.info("\nCalibration table:")
    logger.info(calib_df.to_string(index=False))

    # ========================================================================
    # BASELINE (Point Forecast > 280)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("BASELINE: POINT FORECAST (predictions > 280)")
    logger.info("="*80)

    baseline_results = evaluate_with_prob_threshold(
        all_y_true,
        (all_y_pred_point >= 280).astype(float),  # Convert to prob-like (0 or 1)
        threshold=0.5,  # Dummy threshold since already binary
        site_names=all_site_names
    )

    logger.info(f"\nOVERALL:")
    logger.info(f"  TP={baseline_results['overall']['TP']}, "
                f"FP={baseline_results['overall']['FP']}, "
                f"TN={baseline_results['overall']['TN']}, "
                f"FN={baseline_results['overall']['FN']}")
    logger.info(f"  Sensitivity: {baseline_results['overall']['sensitivity']:.3f}")
    logger.info(f"  Specificity: {baseline_results['overall']['specificity']:.3f}")

    # ========================================================================
    # PROBABILITY THRESHOLD OPTIMIZATION (Multiple Cost Ratios)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PROBABILITY THRESHOLD OPTIMIZATION")
    logger.info("="*80)

    cost_ratios = [5, 10, 20]
    min_threshold = 0.05  # Don't go below 5% probability

    comparison_results = {
        'baseline': baseline_results,
        'calibration': calib_df.to_dict('records'),
        'probability_methods': {}
    }

    for cost_ratio in cost_ratios:
        logger.info(f"\n{'─'*80}")
        logger.info(f"Cost Ratio = {cost_ratio} (FN is {cost_ratio}x more costly than FP)")
        logger.info(f"Minimum threshold = {min_threshold} (5%)")
        logger.info(f"{'─'*80}")

        optimizer = ThresholdOptimizer(
            min_sensitivity=0.5,
            cost_ratio=cost_ratio,
            threshold_step=0.01,
            min_threshold=min_threshold
        )

        optimal = optimizer.optimize(all_y_true_binary, all_y_prob)

        # Evaluate each method
        methods = {
            'youden_j': optimal['youden_j'],
            'cost_weighted': optimal['cost_weighted'],
            'max_specificity': optimal['max_specificity']
        }

        comparison_results['probability_methods'][f'cost_ratio_{cost_ratio}'] = {}

        for method_name, threshold in methods.items():
            results = evaluate_with_prob_threshold(
                all_y_true, all_y_prob, threshold, all_site_names
            )
            results['threshold'] = threshold
            results['method'] = method_name
            results['cost_ratio'] = cost_ratio

            comparison_results['probability_methods'][f'cost_ratio_{cost_ratio}'][method_name] = results

            logger.info(f"\n  {method_name.upper()} (threshold={threshold:.3f}):")
            logger.info(f"    Overall: Sens={results['overall']['sensitivity']:.3f}, "
                       f"Spec={results['overall']['specificity']:.3f}")
            logger.info(f"    TP={results['overall']['TP']}, "
                       f"FP={results['overall']['FP']}, "
                       f"TN={results['overall']['TN']}, "
                       f"FN={results['overall']['FN']}")

    # ========================================================================
    # SUMMARY COMPARISON
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("SUMMARY COMPARISON TABLE")
    logger.info("="*80)

    summary_data = []

    # Baseline
    b = baseline_results['overall']
    summary_data.append({
        'Method': 'Baseline (Point Forecast)',
        'Threshold': '280 (direct)',
        'Cost Ratio': '-',
        'Sensitivity': f"{b['sensitivity']:.3f}",
        'Specificity': f"{b['specificity']:.3f}",
        'TP': b['TP'],
        'FP': b['FP'],
        'FN': b['FN']
    })

    # Probability methods
    for cost_ratio in cost_ratios:
        for method_name in ['youden_j', 'cost_weighted', 'max_specificity']:
            r = comparison_results['probability_methods'][f'cost_ratio_{cost_ratio}'][method_name]['overall']
            thresh = comparison_results['probability_methods'][f'cost_ratio_{cost_ratio}'][method_name]['threshold']
            summary_data.append({
                'Method': f'{method_name} (CR={cost_ratio})',
                'Threshold': f'{thresh:.3f}',
                'Cost Ratio': cost_ratio,
                'Sensitivity': f"{r['sensitivity']:.3f}",
                'Specificity': f"{r['specificity']:.3f}",
                'TP': r['TP'],
                'FP': r['FP'],
                'FN': r['FN']
            })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Save results
    output_path = project_root / "reports" / "threshold_optimization" / "refined_comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=4, default=str)

    logger.info(f"\n\nResults saved to: {output_path}")
    logger.info("\n" + "="*80)
    logger.info("COMPARISON COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
