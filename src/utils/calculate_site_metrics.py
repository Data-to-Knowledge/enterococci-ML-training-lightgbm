"""
Calculate site-specific metrics from TSCV predictions
"""
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
predictions_file = project_root / "reports" / "evaluation_results_fold_predictions.csv"

# Load predictions
df = pd.read_csv(predictions_file)

# Calculate actual and predicted exceedances
actual_exceedance = (df['Enterococci'] >= 280).astype(int)
predicted_exceedance = (df['predictions'] >= 280).astype(int)

# Calculate site-specific metrics
sites = df['SITE_NAME'].unique()
site_metrics = []

for site in sites:
    site_mask = df['SITE_NAME'] == site

    tp = ((predicted_exceedance == 1) & (actual_exceedance == 1) & site_mask).sum()
    fp = ((predicted_exceedance == 1) & (actual_exceedance == 0) & site_mask).sum()
    tn = ((predicted_exceedance == 0) & (actual_exceedance == 0) & site_mask).sum()
    fn = ((predicted_exceedance == 0) & (actual_exceedance == 1) & site_mask).sum()

    sensitivity = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    specificity = (tn / (tn + fp) * 100) if (tn + fp) > 0 else 0

    site_metrics.append({
        'Site': site,
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'Sensitivity %': round(sensitivity, 1),
        'Specificity %': round(specificity, 1)
    })

# Create DataFrame
results_df = pd.DataFrame(site_metrics)
results_df = results_df.sort_values('Site')

# Display site-specific metrics
print('\n' + '='*120)
print('SITE-SPECIFIC METRICS (TSCV)')
print('='*120)
print(results_df.to_string(index=False))

# Calculate overall metrics
overall = {
    'TP': int(results_df['TP'].sum()),
    'FP': int(results_df['FP'].sum()),
    'TN': int(results_df['TN'].sum()),
    'FN': int(results_df['FN'].sum())
}
overall['Sensitivity'] = round(overall['TP'] / (overall['TP'] + overall['FN']) * 100, 1) if (overall['TP'] + overall['FN']) > 0 else 0
overall['Specificity'] = round(overall['TN'] / (overall['TN'] + overall['FP']) * 100, 1) if (overall['TN'] + overall['FP']) > 0 else 0

print('\n' + '='*120)
print('OVERALL METRICS')
print('='*120)
print(f"TP: {overall['TP']}, FP: {overall['FP']}, TN: {overall['TN']}, FN: {overall['FN']}")
print(f"Sensitivity: {overall['Sensitivity']}%, Specificity: {overall['Specificity']}%")
print('='*120 + '\n')
