# Hyperparameter Experiment System

Rapid experimentation framework for testing different hyperparameter combinations while maintaining full traceability between configurations and metrics.

## Quick Start

### 1. Run All Experiments
```bash
python src/experiments/run_hyperparam_experiments.py --compare
```

### 2. Run Specific Experiments
```bash
python src/experiments/run_hyperparam_experiments.py --experiments baseline exp_001_min_leaf exp_002_reg_lambda --compare
```

### 3. View Results Without Re-running
```bash
python src/experiments/view_experiments.py
```

## Experiment Configurations

All experiments are defined in [`src/config/hyperparam_experiments.yaml`](../config/hyperparam_experiments.yaml)

### Current Experiments

| Experiment | Description | Key Changes |
|------------|-------------|-------------|
| `baseline` | Current Round 1 settings | No changes (73% sens, 88.4% spec) |
| `exp_001_min_leaf` | Increased min_data_in_leaf | `min_data_in_leaf: 5 → 10` |
| `exp_002_reg_lambda` | Stronger L2 regularization | `reg_lambda: 0.8 → 1.2` |
| `exp_003_max_depth` | Reduced tree depth | `max_depth: 6 → 5` |
| `exp_004_scale_weight` | Reduced class weight | `scale_pos_weight: 2.0 → 1.5` |
| `exp_005_combined_light` | Light combined tuning | min_leaf=10 + reg_lambda=1.2 |
| `exp_006_combined_moderate` | Moderate combined tuning | min_leaf=10 + reg_lambda=1.2 + max_depth=5 |
| `exp_007_combined_aggressive` | Aggressive combined tuning | All changes combined |
| `exp_008_min_leaf_15` | Very conservative min_leaf | `min_data_in_leaf: 15` |
| `exp_009_reg_lambda_15` | Very strong L2 regularization | `reg_lambda: 1.5` |

## Usage Examples

### Run and Compare Multiple Experiments
```bash
# Test conservative regularization approaches
python src/experiments/run_hyperparam_experiments.py \
  --experiments baseline exp_001_min_leaf exp_002_reg_lambda exp_005_combined_light \
  --compare
```

### View Previous Results
```bash
# View summary of all experiments
python src/experiments/view_experiments.py

# View detailed breakdown
python src/experiments/view_experiments.py --detailed

# View specific experiment
python src/experiments/view_experiments.py --experiment exp_001_min_leaf

# Find best by specificity
python src/experiments/view_experiments.py --best-by specificity

# Find best by sensitivity
python src/experiments/view_experiments.py --best-by sensitivity

# Find best balanced (F1-like)
python src/experiments/view_experiments.py --best-by balanced
```

## Output Structure

```
reports/hyperparam_experiments/
├── baseline_results.json
├── exp_001_min_leaf_results.json
├── exp_002_reg_lambda_results.json
├── ...
├── experiment_comparison.csv          # Summary comparison
└── experiment_comparison_full.csv     # Full metrics
```

### Results Format

Each experiment JSON contains:
```json
{
  "experiment_name": "exp_001_min_leaf",
  "experiment_description": "Test min_data_in_leaf=10...",
  "timestamp": "2025-12-06T10:30:00",
  "hyperparameters": {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "min_data_in_leaf": 10,
    ...
  },
  "metrics": {
    "overall": {
      "sensitivity": 0.730,
      "specificity": 0.901,
      "TP": 108,
      "FP": 115,
      "TN": 1052,
      "FN": 40,
      "rmse": 670.5,
      ...
    },
    "fold_1": {...},
    "fold_2": {...},
    "fold_3": {...}
  }
}
```

## Adding New Experiments

Edit [`src/config/hyperparam_experiments.yaml`](../config/hyperparam_experiments.yaml):

```yaml
experiments:
  exp_010_custom:
    name: "My Custom Experiment"
    description: "Testing custom hyperparameter combination"
    lgb_params:
      n_estimators: 300
      learning_rate: 0.03
      # ... your hyperparameters
```

Then run:
```bash
python src/experiments/run_hyperparam_experiments.py --experiments exp_010_custom
```

## Interpreting Results

### Metrics to Watch

- **Sensitivity**: % of exceedances caught (higher = better, target ≥70%)
- **Specificity**: % of safe samples correct (higher = better, target ≥90%)
- **FP (False Positives)**: Safe samples incorrectly flagged (lower = better)
- **FN (False Negatives)**: Exceedances missed (lower = better, CRITICAL for public health)

### Trade-offs

For water quality forecasting:
- **FN is more costly than FP** (missing contamination is worse than unnecessary warnings)
- Target: Sensitivity ≥70%, Specificity ≥90%
- If choosing between two experiments:
  - Prefer higher sensitivity if specificity difference is small (<2%)
  - Accept 2-3% sensitivity drop for 5%+ specificity gain

### Comparison Example

```
Experiment                     Δ Sens    Δ Spec    Δ FP    Δ FN    Status
exp_001_min_leaf              -0.015    +0.017    -20     +2      ✓ Better
exp_002_reg_lambda            -0.008    +0.012    -15     +1      ✓ Better
exp_005_combined_light        -0.022    +0.025    -30     +3      ⚠ Spec↑ Sens↓
```

- `exp_002_reg_lambda`: Best balance (minimal sensitivity loss, good specificity gain)
- `exp_005_combined_light`: Strong specificity improvement but sensitivity drops below 71%

## Tips

1. **Always run baseline first** to establish reference metrics
2. **Run experiments in batches** of 3-5 to save time
3. **Use --compare flag** to generate comparison report automatically
4. **Check fold-level metrics** to ensure improvements are consistent across time periods
5. **Document winning configuration** by updating main_config.yaml with best experiment's hyperparameters

## Workflow Example

```bash
# Step 1: Run baseline + promising experiments
python src/experiments/run_hyperparam_experiments.py \
  --experiments baseline exp_001_min_leaf exp_002_reg_lambda exp_005_combined_light \
  --compare

# Step 2: Review results
python src/experiments/view_experiments.py --best-by specificity

# Step 3: Test winner's variants
# (Add new experiments to hyperparam_experiments.yaml)
python src/experiments/run_hyperparam_experiments.py \
  --experiments exp_011_variant_a exp_012_variant_b \
  --compare

# Step 4: Update main config with best settings
# (Manually copy hyperparameters from winning experiment to main_config.yaml)
```
