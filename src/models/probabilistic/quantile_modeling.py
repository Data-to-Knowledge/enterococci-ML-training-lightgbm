import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import sys


project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

class ProbabilisticQuantileEnsembleModel:
    """
    Probabilistic Quantile Ensemble Model that trains an ensemble of LightGBM quantile regression models.
    Each model is trained for a specific quantile (e.g., 0.05, 0.30, 0.50, ...).

    This class adheres to the same interface as your existing baseline models, so it integrates seamlessly
    with the main pipeline.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config

        self.target_column = None
        self.feature_columns = None

        self.quantile_levels: List[float] = config.get(
            "quantiles", [0.05, 0.30, 0.50, 0.70, 0.80, 0.90, 0.92, 0.95, 0.98]
        )
        self.params: Dict[str, Any] = config.get("lgb_params", {
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'n_estimators': 100,
            'min_data_in_leaf': 10,
            'lambda_l2': 0.45
        })
        self.models: Dict[float, lgb.LGBMRegressor] = {}

    def train(self, data: pd.DataFrame, *, target_col: str = "Enterococci") -> pd.DataFrame:
        """
        Train all LightGBM quantile models and return an out-of-fold (OOF)
        DataFrame that already includes the target column.

        Parameters
        ----------
        data : pd.DataFrame
            Feature-engineered training set with the target.
        target_col : str, optional
            Column name of the target.  Defaults to "Enterococci".
        """
        self.target_column = target_col

        X = data.drop(columns=[target_col]).copy()
        y = data[target_col].copy()

        # 1. Fit a LightGBM quantile model per quantile on the full data
        for q in self.quantile_levels:
            self.logger.info("fitting LightGBM for q=%.3f", q)
            params = {**self.params,
                    "objective": "quantile",
                    "alpha":     q,
                    "verbose":  -1}
            gbm = lgb.LGBMRegressor(**params)
            gbm.fit(X, y)
            self.models[q] = gbm

        # 2. Build out-of-fold predictions (needed by the meta-learner)
        from sklearn.model_selection import KFold

        n_splits = 2
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_preds = pd.DataFrame(index=data.index)

        for q in self.quantile_levels:
            col = f"q_{q}"
            oof_col = np.full(len(data), np.nan)

            params = {**self.params,
                    "objective": "quantile",
                    "alpha":     q,
                    "verbose":  -1}

            for tr_idx, val_idx in kf.split(X):
                gbm = lgb.LGBMRegressor(**params)
                gbm.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                oof_col[val_idx] = gbm.predict(X.iloc[val_idx])

            # any rows never used as a validation fold -> fall back to full-data model
            nan_mask = np.isnan(oof_col)
            if nan_mask.any():
                oof_col[nan_mask] = self.models[q].predict(X.iloc[nan_mask])

            oof_preds[col] = oof_col

        # 3. Package OOF preds together with the target and return
        oof_preds[target_col] = y.values
        return oof_preds


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions from each quantile model and return a DataFrame
        where each column corresponds to a quantile prediction.
        """
        if not self.models:
            self.logger.error("Attempted to predict before training the ensemble.")
            raise ValueError("No trained models available. Call train() first.")

        predictions = {}
        for quantile, model in self.models.items():
            col_name = f"q_{quantile}"
            predictions[col_name] = model.predict(X)
        return pd.DataFrame(predictions, index=X.index)

    def save(self, output_path: Path) -> None:
        self.logger.info(f"Saving Probabilistic Quantile Ensemble Model to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        self.logger.info("Model saved successfully.")

    @classmethod
    def load(cls, input_path: Path) -> "ProbabilisticQuantileEnsembleModel":
        model = joblib.load(input_path)
        return model
