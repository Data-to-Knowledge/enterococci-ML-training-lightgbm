"""
Test what happens when you wrap a DataFrame in pd.DataFrame()
"""
import pandas as pd
import numpy as np

# Simulate what the probabilistic model returns
prob_model_output = pd.DataFrame({
    'q_0.05': [10, 20, 30],
    'q_0.5': [50, 100, 150],
    'q_0.95': [90, 180, 270],
    'predictions': [52, 102, 152],
    'prob_exceed_280': [0.1, 0.3, 0.5]
})

print("="*80)
print("ORIGINAL DataFrame (from probabilistic model)")
print("="*80)
print(prob_model_output)
print(f"\nType: {type(prob_model_output)}")
print(f"Columns: {prob_model_output.columns.tolist()}")

# Now wrap it in pd.DataFrame() like the time_series_evaluator does
wrapped = pd.DataFrame(prob_model_output)

print("\n" + "="*80)
print("After pd.DataFrame(original)")
print("="*80)
print(wrapped)
print(f"\nType: {type(wrapped)}")
print(f"Columns: {wrapped.columns.tolist()}")
print(f"\nAre they the same? {wrapped.equals(prob_model_output)}")

# Test accessing columns
print("\n" + "="*80)
print("Column Access Test")
print("="*80)
print(f"Original['predictions']: {prob_model_output['predictions'].values}")
print(f"Wrapped['predictions']: {wrapped['predictions'].values}")

# Now test what happens when you add new columns
wrapped["DateTime"] = pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03'])
wrapped["SITE_NAME"] = ['Site A', 'Site B', 'Site C']
wrapped["Enterococci"] = [100, 300, 50]

print("\n" + "="*80)
print("After adding DateTime, SITE_NAME, Enterococci")
print("="*80)
print(wrapped)
print(f"\nColumns: {wrapped.columns.tolist()}")

# This is what gets passed to the dashboard
print("\n" + "="*80)
print("DASHBOARD RECEIVES THIS")
print("="*80)
print(f"Columns in test_forecast dict: {wrapped.columns.tolist()}")
print(f"\nThe 'predictions' column exists: {'predictions' in wrapped.columns}")
print(f"Values in 'predictions' column: {wrapped['predictions'].values}")
