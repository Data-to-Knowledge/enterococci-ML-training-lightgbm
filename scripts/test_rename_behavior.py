"""
Test pandas rename behavior when column doesn't exist
"""
import pandas as pd

df = pd.DataFrame({
    'q_0.05': [10, 20],
    'q_0.5': [50, 100],
    'q_0.95': [90, 180],
    'predictions': [52, 102]
})

print("Original DataFrame:")
print(df)
print(f"Columns: {df.columns.tolist()}")

# Try to rename a column that exists
df_renamed = df.rename(columns={"predictions": "my_model"})
print("\nAfter renaming 'predictions' to 'my_model':")
print(df_renamed)
print(f"Columns: {df_renamed.columns.tolist()}")

# Try to rename a column that doesn't exist
df2 = pd.DataFrame({
    'q_0.05': [10, 20],
    'q_0.5': [50, 100],
    'q_0.95': [90, 180]
    # NO 'predictions' column!
})

print("\n" + "="*80)
print("DataFrame WITHOUT 'predictions' column:")
print(df2)
print(f"Columns: {df2.columns.tolist()}")

df2_renamed = df2.rename(columns={"predictions": "my_model"})
print("\nAfter trying to rename 'predictions' to 'my_model':")
print(df2_renamed)
print(f"Columns: {df2_renamed.columns.tolist()}")
print("Note: rename silently does nothing if column doesn't exist!")

# Now check what happens when you try to access a renamed column
try:
    print(f"\nAccessing df2_renamed['my_model']:")
    print(df2_renamed['my_model'])
except KeyError as e:
    print(f"KeyError: {e}")
    print("The column doesn't exist!")
