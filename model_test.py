import pandas as pd
import joblib

TRAINING_DATA_PATH = r'C:/atman_python/Dont-Swim-in-Data/data/processed/training_data.csv'
MODEL_PATH = r'C:/atman_python/Dont-Swim-in-Data/models/probabilistic_framework.joblib'

# Load
training_df = pd.read_csv(TRAINING_DATA_PATH)
model = joblib.load(MODEL_PATH)

# Your expected categorical columns
CATEGORICAL_FEATURES = [
    'SITE_NAME', 'Harbour', 'tidal_state', 'Shallowness', 'Soil_type',
    'Landcover_catchment', 'wind_shore_3h', 'wind_shore_6h', 'wind_shore_12h'
]

# 1. Print all dtypes and top 10 unique values for quick scan
print("=== TRAINING DATA DTYPES AND SAMPLE VALUES ===")
for col in training_df.columns:
    print(f"{col}: {training_df[col].dtype} -- sample: {training_df[col].unique()[:5]}")
print("\n")

# 2. Which columns are categorical (likely lost after csv load)
print("=== CATEGORICAL FEATURES, as in code ===")
for col in CATEGORICAL_FEATURES:
    if col in training_df.columns:
        print(f"{col}: dtype={training_df[col].dtype} | nuniques={training_df[col].nunique()}")
print("\n")

# 3. Which columns are float but likely should be int
print("=== CHECK FLOATS THAT COULD BE INT ===")
for col in training_df.select_dtypes("float"):
    vals = training_df[col].dropna().unique()
    is_int = all(float(int(x)) == float(x) for x in vals)
    print(f"{col}: float, all integer? {is_int}, example: {vals[:10]}")
print("\n")

# 4. Print model-side expectations
ensemble = model.quantile_ensemble
trained_model = ensemble.models[0.5]
print("=== MODEL EXPECTATIONS ===")
print("Model Feature Names:", trained_model.booster_.feature_name())
try:
    print("Model Categorical Features:", trained_model.booster_.categorical_feature_name())
except AttributeError:
    print("Model does not expose categorical_feature_name (typical with LGBM sklearn wrapper)")
print("\n")

# 5. Try restoring one categorical dtype for demo (optional, only if you know the levels)
# Example:
# levels = ['Akaroa at main beach', ...]  # fill in
# training_df['SITE_NAME'] = pd.Categorical(training_df['SITE_NAME'], categories=levels)
