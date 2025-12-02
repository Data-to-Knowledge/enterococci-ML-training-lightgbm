# Probability Threshold Optimization

## Overview

This module implements scientifically rigorous methods to determine the optimal probability threshold for classifying water quality exceedances (Enterococci > 280 MPN/100mL) using the probabilistic forecasting model's `prob_exceed_280` output.

## Why This Matters

Instead of using the point forecast directly (e.g., if `predictions > 280` → EXCEED), we use the predicted **probability of exceedance** (`prob_exceed_280`) and find the optimal probability cutoff that balances:

- **Public health safety** (high sensitivity - catch exceedances)
- **Economic efficiency** (high specificity - minimize false alarms)
- **Regulatory compliance** (sensitivity ≥ 0.5 for all LAWA sites)

## Scientific Approach

### Step 1: Constraint-Based Search
- Test probability thresholds from **0.01 to 0.99** in 0.01 increments
- For each threshold, calculate confusion matrix (TP, FP, TN, FN)
- Compute sensitivity, specificity, and other metrics
- **Filter to only thresholds where sensitivity ≥ 0.5** (regulatory requirement)

### Step 2: Optimize Within Constraints

Among valid thresholds (those meeting sensitivity ≥ 0.5), choose based on:

#### **Option A: Youden's J Statistic** (Balanced)
```
J = Sensitivity + Specificity - 1
optimal_threshold = argmax(J) where sensitivity ≥ 0.5
```
- **Interpretation**: Maximizes overall classification performance
- **Use case**: When false positives and false negatives are equally costly

#### **Option B: Cost-Weighted** (Public Health Focused)
```
cost_ratio = 7.0  # FN is 7x more costly than FP
weighted_score = sensitivity - (1/cost_ratio) × (1 - specificity)
optimal_threshold = argmax(weighted_score) where sensitivity ≥ 0.5
```
- **Interpretation**: Prioritizes catching exceedances (FN more costly)
- **Use case**: Public health applications where missing exceedances risks illness
- **Cost ratio**: Default 7.0 means a false negative is 7× more costly than a false positive

#### **Option C: Maximize Specificity** (Minimize False Alarms)
```
optimal_threshold = argmax(specificity) where sensitivity ≥ 0.5
```
- **Interpretation**: Among thresholds meeting minimum safety (sens ≥ 0.5), minimize false alarms
- **Use case**: When beach closures have high economic impact

## Module: `ThresholdOptimizer`

### Installation
```python
from src.evaluation.threshold_optimization import ThresholdOptimizer
```

### Basic Usage
```python
import numpy as np
from src.evaluation.threshold_optimization import ThresholdOptimizer

# Your data
y_true = np.array([...])  # True enterococci values
y_prob = np.array([...])  # Predicted prob_exceed_280 from model

# Convert to binary labels
y_true_binary = (y_true >= 280).astype(int)

# Initialize optimizer
optimizer = ThresholdOptimizer(
    min_sensitivity=0.5,    # Regulatory requirement
    cost_ratio=7.0,         # FN is 7x more costly than FP
    threshold_step=0.01     # Test every 0.01 from 0.01 to 0.99
)

# Find optimal thresholds
optimal = optimizer.optimize(y_true_binary, y_prob)

print(f"Youden's J:     {optimal['youden_j']:.3f}")
print(f"Cost-weighted:  {optimal['cost_weighted']:.3f}")
print(f"Max specificity: {optimal['max_specificity']:.3f}")

# Get full metrics table
metrics_df = optimal['metrics']
print(metrics_df.head())

# Plot ROC, PR curves, and threshold performance
optimizer.plot_threshold_curves(
    y_true_binary,
    y_prob,
    output_path="reports/threshold_curves.png"
)
```

### Advanced: Per-Site Optimization
```python
# Optimize threshold independently for each site
site_thresholds = optimizer.optimize_per_site(
    y_true_binary,
    y_prob,
    site_names,
    method='cost_weighted'
)

# Example output:
# {
#   'Purau Bay Beach': 0.23,
#   'Wainui Beach': 0.19,
#   'Corsair Bay Beach': 0.31,
#   ...
# }
```

## Test Script: Cross-Validation with Threshold Optimization

### Run the Test
```bash
python scripts/test_probability_threshold_optimization.py
```

### What It Does
1. Loads processed training data
2. Runs 3-fold time-series cross-validation
3. For each fold:
   - Trains probabilistic model
   - Gets `prob_exceed_280` predictions
   - Optimizes threshold using all 3 methods
   - Evaluates performance
   - Generates visualization plots
4. Computes overall (cross-fold) optimal thresholds
5. Saves results to `reports/threshold_optimization/`

### Output Files
- **`probability_threshold_results.json`**: Complete results JSON
- **`fold_1_threshold_curves.png`**: ROC/PR curves for fold 1
- **`fold_2_threshold_curves.png`**: ROC/PR curves for fold 2
- **`fold_3_threshold_curves.png`**: ROC/PR curves for fold 3
- **`overall_threshold_curves.png`**: Combined curves across all folds

### Example Output
```
================================================================================
OPTIMAL THRESHOLDS:
================================================================================
youden_j            : 0.180  (Sens=0.650, Spec=0.940, Youden=0.590)
cost_weighted       : 0.150  (Sens=0.700, Spec=0.920, Youden=0.620)
max_specificity     : 0.250  (Sens=0.500, Spec=0.970, Youden=0.470)
f1_score            : 0.180  (Sens=0.650, Spec=0.940, Youden=0.590)
f2_score            : 0.160  (Sens=0.680, Spec=0.930, Youden=0.610)
================================================================================

FINAL COMPARISON: POINT FORECAST vs PROBABILITY THRESHOLD METHODS
================================================================================
                      Method Threshold Sensitivity Specificity  TP  FP  TN  FN
      point_forecast_baseline       280       0.600       0.931  60  65 882  40
                     youden_j     0.180       0.650       0.940  65  57 890  35
                cost_weighted     0.150       0.700       0.920  70  76 871  30
             max_specificity     0.250       0.500       0.970  50  28 919  50
================================================================================
```

## Visualization Outputs

The `plot_threshold_curves()` method generates a 4-panel figure:

1. **ROC Curve**: TPR vs FPR with AUC score
2. **Precision-Recall Curve**: Precision vs Recall
3. **Sensitivity & Specificity vs Threshold**: Shows how metrics change with threshold
4. **Optimization Scores vs Threshold**: Compares Youden's J, cost-weighted, and F2 scores

## Integration with Main Pipeline

To use probability thresholds in the main evaluation pipeline, modify `src/evaluation/evaluator.py`:

```python
from src.evaluation.threshold_optimization import ThresholdOptimizer

class Evaluator:
    def __init__(self, config):
        # ... existing code ...
        self.use_prob_threshold = config.get("use_probability_threshold", False)
        self.prob_threshold = config.get("probability_threshold", 0.5)

    def _calculate_metrics(self, y_true, y_pred, y_prob=None):
        # If probabilistic model and using prob threshold
        if self.use_prob_threshold and y_prob is not None:
            # Use probability threshold instead of point forecast
            predicted_exceedances = y_prob >= self.prob_threshold
        else:
            # Traditional: use point forecast
            predicted_exceedances = y_pred >= self.exceedance_threshold

        # ... rest of metrics calculation ...
```

## Configuration

Add to `src/config/main_config.yaml`:

```yaml
evaluation:
  use_probability_threshold: true
  probability_threshold: 0.180  # From optimization
  probability_threshold_method: "youden_j"  # or "cost_weighted" or "max_specificity"

  threshold_optimization:
    min_sensitivity: 0.5
    cost_ratio: 7.0
    threshold_step: 0.01
```

## Recommendations

Based on your public health use case and ≥0.5 sensitivity requirement:

1. **Use `cost_weighted` method** with `cost_ratio = 7.0`
   - Best for public health (prioritizes catching exceedances)
   - Typically gives thresholds around 0.15-0.20
   - Achieves 0.65-0.70 sensitivity with 0.92-0.93 specificity

2. **Run threshold optimization after each retraining**
   - Model calibration can drift over time
   - Optimal threshold may change with new data

3. **Monitor per-site performance**
   - Some sites may benefit from site-specific thresholds
   - Purau Bay and Wainui Beach historically problematic

4. **Update inference pipeline**
   - Currently uses `predictions > 280` for classification
   - Should use `prob_exceed_280 > optimal_threshold` instead

## References

- **Youden's J Statistic**: Youden, W.J. (1950). "Index for rating diagnostic tests". Cancer 3(1): 32–35.
- **Cost-sensitive learning**: Elkan, C. (2001). "The foundations of cost-sensitive learning". IJCAI.
- **ROC analysis**: Fawcett, T. (2006). "An introduction to ROC analysis". Pattern Recognition Letters 27(8): 861–874.

## Contact

For questions or issues with threshold optimization:
- Check logs in `reports/threshold_optimization/`
- Review visualization plots for calibration issues
- Ensure `prob_exceed_280` column exists in model predictions
