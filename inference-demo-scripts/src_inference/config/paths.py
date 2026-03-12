# config/paths.py
from pathlib import Path

# Determine the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define common data directories
CONFIG_DIR = PROJECT_ROOT / "src_inference" / "config"
MAIN_CONFIG_PATH = CONFIG_DIR / "main_config.yaml"
TRAINED_MODEL_DIR = PROJECT_ROOT / "trained_models_inference"
TRAINED_MODEL_PATH = TRAINED_MODEL_DIR / "probabilistic_framework.joblib"
MODEL_TXT_PATH = TRAINED_MODEL_DIR / "probabilistic_framework.txt"
TRAINING_DATA_PATH = TRAINED_MODEL_DIR / "training_data" / "training_data.csv"
# --- Inference outputs ---
PREDICTION_STORAGE_PATH = PROJECT_ROOT / "inference_outputs" / "inference_predictions.csv"
REFERENCE_SCHEMA_PATH = TRAINED_MODEL_DIR / "reference_schema.pkl"
