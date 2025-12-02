"""
Compare classification performance using:
1. Median of MONOTONIC quantiles (current baseline: 'predictions' column)
2. Median of RAW quantiles (before isotonic regression)

This shows the impact of isotonic calibration on classification performance.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the fold predictions with all columns
fold_preds = pd.read_csv("reports/evaluation_results_fold_predictions.csv")

print("="*80)
print("COMPARISON: RAW vs MONOTONIC QUANTILE MEDIANS")
print("="*80)
print(f"\nTotal samples: {len(fold_preds)}")
print(f"Actual exceedances (>=280): {(fold_preds['Enterococci'] >= 280).sum()}")
print(f"Actual safe (<280): {(fold_preds['Enterococci'] < 280).sum()}")

# Get raw quantile columns (exclude q_0.05 as it's not in the saved file based on the column list)
raw_quantile_cols = [col for col in fold_preds.columns if col.endswith('_raw')]
print(f"\nRaw quantile columns: {raw_quantile_cols}")

# Get monotonic quantile columns
mono_quantile_cols = [col for col in fold_preds.columns
                      if col.startswith('q_') and not col.endswith('_raw')]
print(f"Monotonic quantile columns: {mono_quantile_cols}")

# Calculate median of raw quantiles
if raw_quantile_cols:
    median_raw = fold_preds[raw_quantile_cols].median(axis=1)
    print(f"\nMedian of raw quantiles calculated from {len(raw_quantile_cols)} columns")
else:
    print("\nWARNING: No raw quantile columns found!")
    median_raw = None

# The 'predictions' column is already the median of monotonic quantiles
median_mono = fold_preds['predictions']
print(f"Using existing 'predictions' column (median of {len(mono_quantile_cols)} monotonic quantiles)")

# True labels
y_true = fold_preds['Enterococci']
y_true_binary = (y_true >= 280).astype(int)

def calculate_confusion_matrix(y_true, y_pred, threshold=280):
    """Calculate confusion matrix metrics."""
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)

    TP = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
    FP = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
    TN = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
    FN = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return {
        'TP': int(TP),
        'FP': int(FP),
        'TN': int(TN),
        'FN': int(FN),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy
    }

print("\n" + "="*80)
print("BASELINE: Median of MONOTONIC quantiles (current 'predictions' column)")
print("="*80)
mono_metrics = calculate_confusion_matrix(y_true, median_mono)
print(f"TP={mono_metrics['TP']}, FP={mono_metrics['FP']}, "
      f"TN={mono_metrics['TN']}, FN={mono_metrics['FN']}")
print(f"Sensitivity: {mono_metrics['sensitivity']:.3f}")
print(f"Specificity: {mono_metrics['specificity']:.3f}")
print(f"Accuracy: {mono_metrics['accuracy']:.3f}")

if median_raw is not None:
    print("\n" + "="*80)
    print("COMPARISON: Median of RAW quantiles (before isotonic regression)")
    print("="*80)
    raw_metrics = calculate_confusion_matrix(y_true, median_raw)
    print(f"TP={raw_metrics['TP']}, FP={raw_metrics['FP']}, "
          f"TN={raw_metrics['TN']}, FN={raw_metrics['FN']}")
    print(f"Sensitivity: {raw_metrics['sensitivity']:.3f}")
    print(f"Specificity: {raw_metrics['specificity']:.3f}")
    print(f"Accuracy: {raw_metrics['accuracy']:.3f}")

    print("\n" + "="*80)
    print("IMPACT OF ISOTONIC REGRESSION")
    print("="*80)
    print(f"Change in TP: {mono_metrics['TP'] - raw_metrics['TP']:+d}")
    print(f"Change in FP: {mono_metrics['FP'] - raw_metrics['FP']:+d}")
    print(f"Change in TN: {mono_metrics['TN'] - raw_metrics['TN']:+d}")
    print(f"Change in FN: {mono_metrics['FN'] - raw_metrics['FN']:+d}")
    print(f"Change in Sensitivity: {mono_metrics['sensitivity'] - raw_metrics['sensitivity']:+.3f}")
    print(f"Change in Specificity: {mono_metrics['specificity'] - raw_metrics['specificity']:+.3f}")
    print(f"Change in Accuracy: {mono_metrics['accuracy'] - raw_metrics['accuracy']:+.3f}")

    # Check how many predictions differ
    diff_count = (median_mono != median_raw).sum()
    print(f"\nPredictions that differ: {diff_count}/{len(fold_preds)} "
          f"({100*diff_count/len(fold_preds):.1f}%)")

    # Show distribution of differences
    diff = median_mono - median_raw
    print(f"Mean difference (mono - raw): {diff.mean():.2f}")
    print(f"Median difference: {diff.median():.2f}")
    print(f"Max absolute difference: {diff.abs().max():.2f}")

    # How many predictions changed from <280 to ≥280 or vice versa?
    raw_exceedance = median_raw >= 280
    mono_exceedance = median_mono >= 280
    changed = raw_exceedance != mono_exceedance
    print(f"\nPredictions that crossed the 280 threshold: {changed.sum()} "
          f"({100*changed.sum()/len(fold_preds):.1f}%)")

    # Of those that changed, how many were correct?
    if changed.sum() > 0:
        true_exceedance = y_true >= 280
        # Changed from safe to exceedance (raw<280, mono≥280)
        safe_to_exc = (~raw_exceedance) & mono_exceedance
        # Changed from exceedance to safe (raw≥280, mono<280)
        exc_to_safe = raw_exceedance & (~mono_exceedance)

        print(f"  Safe -> Exceedance: {safe_to_exc.sum()}")
        print(f"    Correct (were true exceedances): {(safe_to_exc & true_exceedance).sum()}")
        print(f"    Incorrect (were actually safe): {(safe_to_exc & ~true_exceedance).sum()}")

        print(f"  Exceedance -> Safe: {exc_to_safe.sum()}")
        print(f"    Correct (were actually safe): {(exc_to_safe & ~true_exceedance).sum()}")
        print(f"    Incorrect (were true exceedances): {(exc_to_safe & true_exceedance).sum()}")

print("\n" + "="*80)
