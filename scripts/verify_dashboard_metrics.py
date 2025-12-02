"""
Verify what metrics the dashboard is calculating vs what's in the CSV.
This script simulates the exact logic used by the dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path

def convert_target_to_flag(y, threshold):
    """Convert numeric values to 'EXCEEDANCE' or 'SAFE' based on threshold"""
    return pd.Series(["EXCEEDANCE" if val >= threshold else "SAFE" for val in y])

def calculate_confusion_matrix_from_flags(y_true, y_pred):
    """Calculate confusion matrix from SAFE/EXCEEDANCE flags"""
    conf_matrix = confusion_matrix(y_true, y_pred, labels=["SAFE", "EXCEEDANCE"])

    # Handle case where confusion matrix is 1x1 (only one class present)
    if conf_matrix.shape == (1, 1):
        if y_true.iloc[0] == "SAFE":
            tn, fp, fn, tp = conf_matrix[0, 0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, conf_matrix[0, 0]
    else:
        # Normal case with both classes
        tn, fp, fn, tp = conf_matrix.ravel()

    return tn, fp, fn, tp

# Load the fold predictions CSV (this is what gets saved)
csv_path = Path("reports/evaluation_results_fold_predictions.csv")
df_csv = pd.read_csv(csv_path)

print("=" * 80)
print("VERIFICATION: Dashboard Metrics vs CSV Metrics")
print("=" * 80)
print()

# Method 1: Direct calculation from CSV (what we've been doing)
print("METHOD 1: Direct calculation from CSV")
print("-" * 80)
threshold = 280
y_true_numeric = df_csv['Enterococci']
y_pred_numeric = df_csv['predictions']

y_true_binary = (y_true_numeric >= threshold).astype(int)
y_pred_binary = (y_pred_numeric >= threshold).astype(int)

TP = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
FP = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
TN = ((y_true_binary == 0) & (y_pred_binary == 0)).sum()
FN = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()

print(f"Total rows: {len(df_csv)}")
print(f"TP = {TP}")
print(f"FP = {FP}")
print(f"TN = {TN}")
print(f"FN = {FN}")
print()

# Method 2: Dashboard's approach (convert to flags first, then use sklearn)
print("METHOD 2: Dashboard approach (convert to flags)")
print("-" * 80)

# Simulate what the dashboard does
y_true_flags = convert_target_to_flag(df_csv['Enterococci'], 280)
y_pred_flags = convert_target_to_flag(df_csv['predictions'], 280)

tn_dash, fp_dash, fn_dash, tp_dash = calculate_confusion_matrix_from_flags(y_true_flags, y_pred_flags)

print(f"Total rows: {len(df_csv)}")
print(f"TP = {tp_dash}")
print(f"FP = {fp_dash}")
print(f"TN = {tn_dash}")
print(f"FN = {fn_dash}")
print()

# Check if they match
print("=" * 80)
if TP == tp_dash and FP == fp_dash and TN == tn_dash and FN == fn_dash:
    print("RESULT: Methods match! Dashboard should show the same metrics.")
else:
    print("WARNING: Methods don't match!")
    print(f"Difference in TP: {tp_dash - TP}")
    print(f"Difference in FP: {fp_dash - FP}")
    print(f"Difference in TN: {tn_dash - TN}")
    print(f"Difference in FN: {fn_dash - FN}")

print("=" * 80)
print()

# Now check what the JSON file says
import json
json_path = Path("reports/evaluation_results.json")
with open(json_path, 'r') as f:
    json_data = json.load(f)

print("METHOD 3: What's stored in evaluation_results.json")
print("-" * 80)
overall = json_data['results']['fold_results']['overall']
print(f"TP = {overall['TP']}")
print(f"FP = {overall['FP']}")
print(f"TN = {overall['TN']}")
print(f"FN = {overall['FN']}")
print(f"Sensitivity = {overall['sensitivity']:.4f}")
print(f"Specificity = {overall['specificity']:.4f}")
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"CSV Direct Calculation:  TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"Dashboard Method:        TP={tp_dash}, FP={fp_dash}, TN={tn_dash}, FN={fn_dash}")
print(f"JSON File:               TP={overall['TP']}, FP={overall['FP']}, TN={overall['TN']}, FN={overall['FN']}")
print()

# Check site counts
print("=" * 80)
print("SITE BREAKDOWN")
print("=" * 80)
sites = df_csv['SITE_NAME'].unique()
print(f"Total unique sites in CSV: {len(sites)}")
print(f"Sites: {sorted(sites)}")
print()

# Check if any sites were excluded
excluded_sites = [
    'Scarborough Beach by clock tower',
    'Sumner Beach Surf club',
    'Caroline Bay - mid beach',
    'Timaru Coast Caroline Bay at Virtue Avenue',
    'Timaru Coast at yacht club jetty',
    'Taylors Mistake Beach Surf club',
    'Governors Bay Sandy Beach',
    'Cass Bay at boat ramp'
]

print("Sites that SHOULD be excluded:")
for site in excluded_sites:
    if site in sites:
        print(f"  WARNING: {site} is present in CSV (should be excluded!)")
    else:
        print(f"  OK: {site} is NOT in CSV (correctly excluded)")

print("=" * 80)
