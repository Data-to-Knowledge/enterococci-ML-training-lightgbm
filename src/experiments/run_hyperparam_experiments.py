"""
Hyperparameter Experiment Runner

Runs multiple hyperparameter configurations and tracks metrics for each experiment.
Results are saved to reports/hyperparam_experiments/ with full traceability.

Usage:
    # Run all experiments
    python src/experiments/run_hyperparam_experiments.py

    # Run specific experiments
    python src/experiments/run_hyperparam_experiments.py --experiments baseline exp_001_min_leaf exp_002_reg_lambda

    # Run and generate comparison report
    python src/experiments/run_hyperparam_experiments.py --compare
"""

import os
import sys
import yaml
import json
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.main_pipeline import run_pipeline


def load_experiments():
    """Load experiment configurations from YAML file."""
    exp_config_path = project_root / "src" / "config" / "hyperparam_experiments.yaml"

    with open(exp_config_path, 'r') as f:
        exp_config = yaml.safe_load(f)

    return exp_config['experiments']


def run_single_experiment(exp_name, exp_config, base_config):
    """Run a single experiment with specified hyperparameters."""

    print(f"\n{'='*120}")
    print(f"RUNNING EXPERIMENT: {exp_name}")
    print(f"Description: {exp_config['description']}")
    print(f"{'='*120}\n")

    # Update config with experiment hyperparameters
    config = base_config.copy()
    config['models']['probabilistic_framework']['lgb_params'] = exp_config['lgb_params']

    # Create temporary config file for this experiment
    temp_config_file = project_root / "src" / "config" / f"temp_exp_{exp_name}.yaml"

    with open(temp_config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run pipeline with updated config
    print("Running pipeline...")
    try:
        run_pipeline(str(temp_config_file), mode="evaluate")
    finally:
        # Clean up temp config file
        if temp_config_file.exists():
            temp_config_file.unlink()

    # Load results
    results_path = project_root / "reports" / "evaluation_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Extract key metrics
    overall = results['results']['fold_results']['overall']
    fold_1 = results['results']['fold_results']['fold_1']
    fold_2 = results['results']['fold_results']['fold_2']
    fold_3 = results['results']['fold_results']['fold_3']

    # Package experiment results
    experiment_results = {
        'experiment_name': exp_name,
        'experiment_description': exp_config['description'],
        'hyperparameters': exp_config['lgb_params'],
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'overall': {
                'sensitivity': overall['sensitivity'],
                'specificity': overall['specificity'],
                'TP': overall['TP'],
                'FP': overall['FP'],
                'TN': overall['TN'],
                'FN': overall['FN'],
                'rmse': overall['rmse'],
                'rmse_safe': overall['rmse_safe'],
                'rmse_exceedance': overall['rmse_exceedance']
            },
            'fold_1': {
                'sensitivity': fold_1['sensitivity'],
                'specificity': fold_1['specificity'],
                'TP': fold_1['TP'],
                'FP': fold_1['FP'],
                'FN': fold_1['FN']
            },
            'fold_2': {
                'sensitivity': fold_2['sensitivity'],
                'specificity': fold_2['specificity'],
                'TP': fold_2['TP'],
                'FP': fold_2['FP'],
                'FN': fold_2['FN']
            },
            'fold_3': {
                'sensitivity': fold_3['sensitivity'],
                'specificity': fold_3['specificity'],
                'TP': fold_3['TP'],
                'FP': fold_3['FP'],
                'FN': fold_3['FN']
            }
        }
    }

    # Save individual experiment results
    exp_dir = project_root / "reports" / "hyperparam_experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_file = exp_dir / f"{exp_name}_results.json"
    with open(exp_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\n[OK] Experiment results saved to: {exp_file}")
    print(f"  Sensitivity: {overall['sensitivity']:.3f}")
    print(f"  Specificity: {overall['specificity']:.3f}")
    print(f"  FP: {overall['FP']}, FN: {overall['FN']}")

    return experiment_results


def generate_comparison_report(experiment_results):
    """Generate comparison report across all experiments."""

    if not experiment_results:
        print("\n[WARN] No experiment results to compare.")
        return None

    print(f"\n{'='*120}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*120}\n")

    # Create comparison DataFrame
    rows = []
    for result in experiment_results:
        metrics = result['metrics']['overall']
        rows.append({
            'Experiment': result['experiment_name'],
            'Description': result['experiment_description'],
            'Sensitivity': metrics['sensitivity'],
            'Specificity': metrics['specificity'],
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'TN': metrics['TN'],
            'FN': metrics['FN'],
            'RMSE': metrics['rmse'],
            'RMSE_Safe': metrics['rmse_safe'],
            'RMSE_Exceed': metrics['rmse_exceedance']
        })

    df = pd.DataFrame(rows)

    # Sort by specificity (descending) then sensitivity (descending)
    df = df.sort_values(['Specificity', 'Sensitivity'], ascending=[False, False])

    # Save to CSV
    exp_dir = project_root / "reports" / "hyperparam_experiments"
    csv_file = exp_dir / "experiment_comparison.csv"
    df.to_csv(csv_file, index=False)

    # Display comparison table
    print("\nEXPERIMENT COMPARISON (sorted by Specificity, then Sensitivity)")
    print("-" * 120)
    print(df[['Experiment', 'Sensitivity', 'Specificity', 'FP', 'FN']].to_string(index=False))
    print()

    # Calculate deltas from baseline
    baseline_row = df[df['Experiment'] == 'baseline']
    if len(baseline_row) > 0:
        baseline_sens = baseline_row['Sensitivity'].values[0]
        baseline_spec = baseline_row['Specificity'].values[0]
        baseline_fp = baseline_row['FP'].values[0]
        baseline_fn = baseline_row['FN'].values[0]

        print("\nDELTA FROM BASELINE")
        print("-" * 120)
        print(f"{'Experiment':<30} {'Delta Sens':<15} {'Delta Spec':<15} {'Delta FP':<10} {'Delta FN':<10}")
        print("-" * 120)

        for _, row in df.iterrows():
            if row['Experiment'] == 'baseline':
                continue

            delta_sens = row['Sensitivity'] - baseline_sens
            delta_spec = row['Specificity'] - baseline_spec
            delta_fp = int(row['FP'] - baseline_fp)
            delta_fn = int(row['FN'] - baseline_fn)

            print(f"{row['Experiment']:<30} {delta_sens:+.3f}          {delta_spec:+.3f}          {delta_fp:+d}        {delta_fn:+d}")

    # Save detailed report
    report_file = exp_dir / "experiment_comparison_full.csv"
    df.to_csv(report_file, index=False)

    print(f"\n[OK] Full comparison saved to: {report_file}")
    print(f"[OK] Summary saved to: {csv_file}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter experiments")
    parser.add_argument('--experiments', nargs='+', help='Specific experiments to run (default: all)')
    parser.add_argument('--compare', action='store_true', help='Generate comparison report after running')

    args = parser.parse_args()

    # Load base config
    base_config_path = project_root / "src" / "config" / "main_config.yaml"
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Load experiments
    experiments = load_experiments()

    # Filter experiments if specified
    if args.experiments:
        experiments = {k: v for k, v in experiments.items() if k in args.experiments}
        print(f"Running {len(experiments)} specified experiments: {list(experiments.keys())}")
    else:
        print(f"Running all {len(experiments)} experiments")

    # Run experiments
    experiment_results = []
    for exp_name, exp_config in experiments.items():
        try:
            result = run_single_experiment(exp_name, exp_config, base_config)
            experiment_results.append(result)
        except Exception as e:
            print(f"\n[FAIL] Experiment {exp_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate comparison report if requested or if multiple experiments ran
    if args.compare or len(experiment_results) > 1:
        generate_comparison_report(experiment_results)

    print(f"\n{'='*120}")
    print(f"COMPLETED {len(experiment_results)}/{len(experiments)} EXPERIMENTS")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()
