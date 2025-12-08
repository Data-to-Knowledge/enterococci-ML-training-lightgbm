"""
View and Compare Hyperparameter Experiment Results

Quickly view results from previous experiments without re-running them.

Usage:
    # View all experiment results
    python src/experiments/view_experiments.py

    # View specific experiment
    python src/experiments/view_experiments.py --experiment exp_001_min_leaf

    # Show detailed per-fold breakdown
    python src/experiments/view_experiments.py --detailed

    # Find best experiments by metric
    python src/experiments/view_experiments.py --best-by specificity
    python src/experiments/view_experiments.py --best-by sensitivity
"""

import json
import argparse
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
exp_dir = project_root / "reports" / "hyperparam_experiments"


def load_all_experiments():
    """Load all experiment results from JSON files."""
    if not exp_dir.exists():
        print(f"No experiments found. Directory does not exist: {exp_dir}")
        return []

    experiments = []
    for exp_file in exp_dir.glob("*_results.json"):
        with open(exp_file, 'r') as f:
            experiments.append(json.load(f))

    return experiments


def display_summary(experiments):
    """Display summary comparison of all experiments."""
    if not experiments:
        print("No experiment results found.")
        return

    print(f"\n{'='*120}")
    print(f"HYPERPARAMETER EXPERIMENT SUMMARY ({len(experiments)} experiments)")
    print(f"{'='*120}\n")

    rows = []
    for exp in experiments:
        metrics = exp['metrics']['overall']
        rows.append({
            'Experiment': exp['experiment_name'],
            'Sensitivity': metrics['sensitivity'],
            'Specificity': metrics['specificity'],
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'TN': metrics['TN'],
            'FN': metrics['FN'],
            'RMSE': round(metrics['rmse'], 1)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(['Specificity', 'Sensitivity'], ascending=[False, False])

    print(df.to_string(index=False))

    # Find baseline if exists
    baseline = df[df['Experiment'] == 'baseline']
    if len(baseline) > 0:
        print(f"\n{'='*120}")
        print("IMPROVEMENTS OVER BASELINE")
        print(f"{'='*120}\n")

        baseline_sens = baseline['Sensitivity'].values[0]
        baseline_spec = baseline['Specificity'].values[0]
        baseline_fp = baseline['FP'].values[0]
        baseline_fn = baseline['FN'].values[0]

        print(f"{'Experiment':<30} {'Delta Sens':<12} {'Delta Spec':<12} {'Delta FP':<10} {'Delta FN':<10} {'Status':<20}")
        print("-" * 120)

        for _, row in df.iterrows():
            if row['Experiment'] == 'baseline':
                continue

            delta_sens = row['Sensitivity'] - baseline_sens
            delta_spec = row['Specificity'] - baseline_spec
            delta_fp = int(row['FP'] - baseline_fp)
            delta_fn = int(row['FN'] - baseline_fn)

            # Determine status
            if delta_spec > 0 and delta_sens >= -0.02:
                status = "✓ Better"
            elif delta_spec > 0 and delta_sens < -0.02:
                status = "⚠ Spec↑ Sens↓"
            elif delta_spec < 0 and delta_sens > 0:
                status = "⚠ Sens↑ Spec↓"
            else:
                status = "✗ Worse"

            print(f"{row['Experiment']:<30} {delta_sens:+.3f}       {delta_spec:+.3f}       {delta_fp:+4d}      {delta_fn:+4d}      {status:<20}")


def display_detailed(experiments):
    """Display detailed per-fold breakdown for all experiments."""
    if not experiments:
        print("No experiment results found.")
        return

    print(f"\n{'='*120}")
    print("DETAILED PER-FOLD BREAKDOWN")
    print(f"{'='*120}\n")

    for exp in experiments:
        print(f"\n{exp['experiment_name']}: {exp['experiment_description']}")
        print("-" * 120)

        # Overall
        overall = exp['metrics']['overall']
        print(f"OVERALL: Sens={overall['sensitivity']:.3f}, Spec={overall['specificity']:.3f}, "
              f"TP={overall['TP']}, FP={overall['FP']}, FN={overall['FN']}")

        # Per-fold
        for fold_name in ['fold_1', 'fold_2', 'fold_3']:
            fold = exp['metrics'][fold_name]
            print(f"  {fold_name}: Sens={fold['sensitivity']:.3f}, Spec={fold['specificity']:.3f}, "
                  f"TP={fold['TP']}, FP={fold['FP']}, FN={fold['FN']}")


def display_single_experiment(experiment_name, experiments):
    """Display details for a single experiment."""
    exp = next((e for e in experiments if e['experiment_name'] == experiment_name), None)

    if not exp:
        print(f"Experiment '{experiment_name}' not found.")
        return

    print(f"\n{'='*120}")
    print(f"EXPERIMENT: {exp['experiment_name']}")
    print(f"{'='*120}\n")

    print(f"Description: {exp['experiment_description']}")
    print(f"Timestamp: {exp['timestamp']}")
    print()

    print("HYPERPARAMETERS:")
    print("-" * 120)
    for param, value in exp['hyperparameters'].items():
        print(f"  {param:<25} {value}")
    print()

    print("METRICS:")
    print("-" * 120)
    overall = exp['metrics']['overall']
    print(f"  Overall Sensitivity:      {overall['sensitivity']:.3f} (TP={overall['TP']}, FN={overall['FN']})")
    print(f"  Overall Specificity:      {overall['specificity']:.3f} (TN={overall['TN']}, FP={overall['FP']})")
    print(f"  RMSE Overall:             {overall['rmse']:.1f}")
    print(f"  RMSE Safe:                {overall['rmse_safe']:.1f}")
    print(f"  RMSE Exceedance:          {overall['rmse_exceedance']:.1f}")
    print()

    print("PER-FOLD METRICS:")
    print("-" * 120)
    print(f"{'Fold':<10} {'Sensitivity':<15} {'Specificity':<15} {'TP':<6} {'FP':<6} {'FN':<6}")
    print("-" * 120)

    for fold_name in ['fold_1', 'fold_2', 'fold_3']:
        fold = exp['metrics'][fold_name]
        print(f"{fold_name:<10} {fold['sensitivity']:<15.3f} {fold['specificity']:<15.3f} "
              f"{fold['TP']:<6} {fold['FP']:<6} {fold['FN']:<6}")


def find_best_by_metric(experiments, metric):
    """Find best experiments by specified metric."""
    if not experiments:
        print("No experiment results found.")
        return

    print(f"\n{'='*120}")
    print(f"TOP 5 EXPERIMENTS BY {metric.upper()}")
    print(f"{'='*120}\n")

    rows = []
    for exp in experiments:
        metrics = exp['metrics']['overall']
        rows.append({
            'Experiment': exp['experiment_name'],
            'Sensitivity': metrics['sensitivity'],
            'Specificity': metrics['specificity'],
            'FP': metrics['FP'],
            'FN': metrics['FN'],
            'RMSE': round(metrics['rmse'], 1)
        })

    df = pd.DataFrame(rows)

    if metric == 'sensitivity':
        df = df.sort_values('Sensitivity', ascending=False)
    elif metric == 'specificity':
        df = df.sort_values('Specificity', ascending=False)
    elif metric == 'balanced':
        df['F1'] = 2 * (df['Sensitivity'] * df['Specificity']) / (df['Sensitivity'] + df['Specificity'])
        df = df.sort_values('F1', ascending=False)
    else:
        print(f"Unknown metric: {metric}")
        return

    print(df.head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="View hyperparameter experiment results")
    parser.add_argument('--experiment', help='View specific experiment details')
    parser.add_argument('--detailed', action='store_true', help='Show detailed per-fold breakdown')
    parser.add_argument('--best-by', choices=['sensitivity', 'specificity', 'balanced'],
                        help='Find best experiments by metric')

    args = parser.parse_args()

    # Load all experiments
    experiments = load_all_experiments()

    if args.experiment:
        display_single_experiment(args.experiment, experiments)
    elif args.detailed:
        display_detailed(experiments)
    elif args.best_by:
        find_best_by_metric(experiments, args.best_by)
    else:
        display_summary(experiments)


if __name__ == "__main__":
    main()
