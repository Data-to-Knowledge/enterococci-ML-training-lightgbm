"""
Diagnose what data is being passed to the dashboard.
"""
import pandas as pd
import numpy as np

# Load the saved fold predictions
fold_preds = pd.read_csv("reports/evaluation_results_fold_predictions.csv")

print("="*80)
print("SAVED FOLD PREDICTIONS CSV")
print("="*80)
print(f"Shape: {fold_preds.shape}")
print(f"Columns: {fold_preds.columns.tolist()}")
print(f"\nTotal samples: {len(fold_preds)}")

# Calculate confusion matrix from saved CSV
y_true = fold_preds['Enterococci']
y_pred = fold_preds['predictions']

y_true_binary = (y_true >= 280).astype(int)
y_pred_binary = (y_pred >= 280).astype(int)

TP = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
FP = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
TN = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
FN = np.sum((y_pred_binary == 0) & (y_true_binary == 1))

print(f"\nMetrics from saved CSV (predictions >= 280):")
print(f"TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"Sensitivity={TP/(TP+FN):.3f}, Specificity={TN/(TN+FP):.3f}")

# Now check what the dashboard would see if it used q_median instead
print("\n" + "="*80)
print("HYPOTHESIS: Dashboard might be using a different column")
print("="*80)

# Simulate what would happen if the full DataFrame was passed with q_median
# For this we need to load or simulate the full predictions
print("\nIf the dashboard received the full DataFrame with q_median column,")
print("and used q_median instead of predictions, the metrics would differ.")

# Let's calculate what TP/FP/TN/FN would need to be to get 62/62/765/23
print("\n" + "="*80)
print("USER'S DASHBOARD SHOWS:")
print("="*80)
print("TP=62, FP=62, TN=765, FN=23")
print(f"Total: {62+62+765+23} samples")
print(f"Sensitivity={62/(62+23):.3f}, Specificity={765/(765+62):.3f}")

# Compare
print("\n" + "="*80)
print("DISCREPANCY ANALYSIS")
print("="*80)
print(f"Difference in TP: {62-TP} (dashboard has {62-TP} more true positives)")
print(f"Difference in FP: {62-FP} (dashboard has {62-FP} more false positives)")
print(f"Difference in TN: {765-TN} (dashboard has {765-TN} fewer true negatives)")
print(f"Difference in FN: {23-FN} (dashboard has {23-FN} fewer false negatives)")

# The sum of differences should be zero (same total samples)
print(f"\nTotal difference: {(62-TP) + (62-FP) + (765-TN) + (23-FN)} (should be 0)")

# Check if it's just 4 samples being classified differently
print("\n" + "="*80)
print("PATTERN ANALYSIS")
print("="*80)
print(f"Extra TP in dashboard: {62-TP}")
print(f"Missing FN in dashboard: {27-23} = {27-23}")
print(f"Extra FP in dashboard: {62-FP}")
print(f"Missing TN in dashboard: {772-765} = {772-765}")

print("\nThis suggests that:")
print("- 4 samples moved from FN to TP (4 actual exceedances now predicted as exceedances)")
print("- 7 samples moved from TN to FP (7 actual safe now predicted as exceedances)")
print("\nThis means the dashboard is using HIGHER predictions than what's saved!")
