import joblib
import pandas as pd


# Path to your saved model
model = joblib.load("C:/atman_python/Dont-Swim-in-Data/models/probabilistic_framework_v2.joblib")
MODEL_TXT_PATH = "C:/atman_python/Dont-Swim-in-Data/models/probabilistic_framework_v2.txt"

booster = model.quantile_ensemble.models[0.5].booster_
booster.save_model(MODEL_TXT_PATH)

print(f"Model saved to: {MODEL_TXT_PATH}")
