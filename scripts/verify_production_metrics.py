"""
Verify production metrics from evaluation_results.json match fold_predictions.csv
"""
import json
import pandas as pd

# Load production results
with open("reports/evaluation_results.json", "r") as f:
    results = json.load(f)

# Check overall metrics
overall = results["results"]["fold_results"]["overall"]
print("Production Evaluation Results (evaluation_results.json):")
print(f"  TP={overall['TP']}, FP={overall['FP']}, TN={overall['TN']}, FN={overall['FN']}")
print(f"  Sensitivity={overall['sensitivity']:.3f}, Specificity={overall['specificity']:.3f}")
print(f"  Total samples: {overall['TP'] + overall['FP'] + overall['TN'] + overall['FN']}")

# Load fold predictions
fold_preds = pd.read_csv("reports/evaluation_results_fold_predictions.csv")
print(f"\nFold predictions CSV: {len(fold_preds)} rows")
print(f"Unique sites: {fold_preds['SITE_NAME'].nunique()}")
print(f"\nSite distribution:")
print(fold_preds['SITE_NAME'].value_counts().sort_index())

# Calculate confusion matrix from fold predictions
y_true = fold_preds["Enterococci"]
y_pred = fold_preds["predictions"]

y_true_binary = (y_true >= 280).astype(int)
y_pred_binary = (y_pred >= 280).astype(int)

TP = ((y_pred_binary == 1) & (y_true_binary == 1)).sum()
FP = ((y_pred_binary == 1) & (y_true_binary == 0)).sum()
TN = ((y_pred_binary == 0) & (y_true_binary == 0)).sum()
FN = ((y_pred_binary == 0) & (y_true_binary == 1)).sum()

sens = TP / (TP + FN) if (TP + FN) > 0 else 0
spec = TN / (TN + FP) if (TN + FP) > 0 else 0

print(f"\nRecalculated from fold_predictions.csv:")
print(f"  TP={TP}, FP={FP}, TN={TN}, FN={FN}")
print(f"  Sensitivity={sens:.3f}, Specificity={spec:.3f}")
print(f"  Total: {TP + FP + TN + FN}")

print("\n" + "="*80)
if TP == overall['TP'] and FN == overall['FN']:
    print("✓ MATCH: Metrics are consistent!")
else:
    print("✗ MISMATCH: Metrics differ!")
    print(f"  Difference: TP delta={TP - overall['TP']}, FN delta={FN - overall['FN']}")
