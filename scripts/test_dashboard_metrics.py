"""
Test what metrics the dashboard would show by replicating its calculation logic
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the fold predictions that the dashboard uses
fold_preds = pd.read_csv("reports/evaluation_results_fold_predictions.csv")

print(f"Total rows: {len(fold_preds)}")
print(f"Unique sites: {fold_preds['SITE_NAME'].nunique()}")

# Replicate dashboard's convert_target_to_flag logic
def convert_target_to_flag(y, threshold):
    """Convert target values to binary flags based on threshold."""
    return (y > threshold).astype(int)

# Replicate dashboard calculation
y_true = fold_preds["Enterococci"]
y_pred = fold_preds["predictions"]

# Dashboard uses > not >=
y_true_flag = convert_target_to_flag(y_true, 280)
y_pred_flag = convert_target_to_flag(y_pred, 280)

TP = np.sum((y_pred_flag == 1) & (y_true_flag == 1))
FP = np.sum((y_pred_flag == 1) & (y_true_flag == 0))
TN = np.sum((y_pred_flag == 0) & (y_true_flag == 0))
FN = np.sum((y_pred_flag == 0) & (y_true_flag == 1))

sens = TP / (TP + FN) if (TP + FN) > 0 else 0
spec = TN / (TN + FP) if (TN + FP) > 0 else 0

print(f"\nDashboard method (y > 280):")
print(f"  TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"  Sensitivity={sens:.3f}, Specificity={spec:.3f}")

# Also try with >=
y_true_flag_ge = (y_true >= 280).astype(int)
y_pred_flag_ge = (y_pred >= 280).astype(int)

TP_ge = np.sum((y_pred_flag_ge == 1) & (y_true_flag_ge == 1))
FN_ge = np.sum((y_pred_flag_ge == 0) & (y_true_flag_ge == 1))
sens_ge = TP_ge / (TP_ge + FN_ge) if (TP_ge + FN_ge) > 0 else 0

print(f"\nAlternative method (y >= 280):")
print(f"  TP={TP_ge}, FN={FN_ge}, Sensitivity={sens_ge:.3f}")
