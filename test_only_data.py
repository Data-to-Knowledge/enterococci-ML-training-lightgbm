import pandas as pd

# === Load predictions ===
predictions = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\results\predictions.csv")

# === Load the test folds ===
test1 = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\data\processed\test_fold_1.csv")
test2 = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\data\processed\test_fold_2.csv")
test3 = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\data\processed\test_fold_3.csv")

# === Combine test folds ===
test_all = pd.concat([test1, test2, test3], ignore_index=True)

# === Standardize DateTime and site name formats ===
def clean_datetime(df, col="DateTime"):
    df[col] = pd.to_datetime(df[col], errors='coerce').dt.floor("min")
    return df

def clean_sitename(df, col="SITE_NAME"):
    df[col] = df[col].astype(str).str.strip()
    return df

# Apply cleaning
predictions = clean_datetime(predictions)
test_all = clean_datetime(test_all)
predictions = clean_sitename(predictions)
test_all = clean_sitename(test_all)

# === Merge to filter only matching rows ===
filtered = pd.merge(
    test_all[["DateTime", "SITE_NAME"]],
    predictions,
    on=["DateTime", "SITE_NAME"],
    how="inner"
)

# === Check how many expected records are matched ===
expected = len(test_all)
actual = len(filtered)
print(f"Matched {actual} out of {expected} test fold entries.")

# === Save to CSV ===
filtered.to_csv(r"C:\atman_python\Dont-Swim-in-Data\results\predictions_filtered_test_only.csv", index=False)
print("Filtered predictions saved.")



# import pandas as pd

# # Load predictions.csv
# # df = pd.read_csv("results/predictions.csv", parse_dates=["DateTime"])
# df = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\results\predictions.csv", parse_dates=["DateTime"])

# # Define the test set date ranges (same as student's time-series evaluation)
# test_periods = [
#     ("2021-10-01", "2022-10-01"),
#     ("2022-10-01", "2023-10-01"),
#     ("2023-10-01", "2024-10-01")
# ]

# # Convert to datetime
# df["DateTime"] = pd.to_datetime(df["DateTime"])

# # Combine all test periods into a single filtered DataFrame
# test_set = pd.concat([
#     df[(df["DateTime"] >= pd.to_datetime(start)) & (df["DateTime"] < pd.to_datetime(end))]
#     for start, end in test_periods
# ], axis=0)

# # Optional: Save it if needed
# test_set.to_csv(r"C:\atman_python\Dont-Swim-in-Data\results\predictions_test_set_only.csv", index=False)

# # Sanity check: number of rows and unique sites
# print(f"✅ Filtered test set rows: {len(test_set)}")
# print(f"🔍 Sites in test set: {test_set['SITE_NAME'].nunique()}")


# import pandas as pd

# # === Load the original predictions ===
# predictions = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\results\predictions.csv")

# # === Load the known test folds (from Asif's evaluation) ===
# test1 = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\data\processed\test_fold_1.csv")
# test2 = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\data\processed\test_fold_2.csv")
# test3 = pd.read_csv(r"C:\atman_python\Dont-Swim-in-Data\data\processed\test_fold_3.csv")

# # === Concatenate all known test rows ===
# test_all = pd.concat([test1, test2, test3], ignore_index=True)

# # === Match on DateTime and SITE_NAME ===
# # Ensure consistent formatting of datetime
# predictions["DateTime"] = pd.to_datetime(predictions["DateTime"])
# test_all["DateTime"] = pd.to_datetime(test_all["DateTime"])

# # Merge/join to get only matching rows from predictions
# filtered = pd.merge(
#     test_all[["DateTime", "SITE_NAME"]],
#     predictions,
#     on=["DateTime", "SITE_NAME"],
#     how="inner"
# )

# # === Save to CSV ===
# filtered.to_csv(r"C:\atman_python\Dont-Swim-in-Data\results\predictions_filtered_test_only.csv", index=False)
# print(f"Filtered predictions saved with {len(filtered)} rows.")
