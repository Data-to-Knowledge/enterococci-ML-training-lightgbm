import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import sys
import lightgbm as lgb
import os
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression


project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.models.probabilistic.quantile_modeling import ProbabilisticQuantileEnsembleModel
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class ProbabilisticForecastingModel:
    """
    Probabilistic framework = 3-stage model
      1)  LightGBM quantile ensemble
      2)  (optional) point-forecast stacker ("meta-learner")
      3)  (optional) interval calibration
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger  = logging.getLogger(self.__class__.__name__)
        self.config  = config
        self.pf_cfg  = config["models"]["probabilistic_framework"]

        self.target_column   = None
        self.feature_columns = None

        # stage-1: quantile models
        self.quantile_ensemble = ProbabilisticQuantileEnsembleModel(self.pf_cfg)

        # stage-2: meta-learner switch & object
        self.use_meta_learner = self.pf_cfg.get("meta_learner", False)
        self.meta_learner     = None

        # stage-3: interval calibration placeholders
        self.calibration_params = None

    def train(self, data: pd.DataFrame) -> None:
        """
        Fit the full probabilistic-forecasting pipeline.

        Stage-1  ->  one LightGBM per quantile
        Stage-2  ->  optional LightGBM stacker (meta-learner)

        Parameters
        ----------
        data : pd.DataFrame
            Feature-engineered training set that already contains the target.
        """
        self.logger.info("Training Probabilistic Forecasting Model")

        target_col          = self.config["data"].get("target_column", "Enterococci")
        self.target_column  = target_col

        Path("data/processed").mkdir(parents=True, exist_ok=True)
        data.to_csv("data/processed/training_data.csv", index=False)

        train_df            = data.copy()
        train_df["DateTime"] = 0

        self.training_data  = train_df

        self.logger.info("   Fitting LightGBM quantile ensemble")
        self.training_oof_quantile_forecast = self.quantile_ensemble.train(
            train_df,
            target_col=target_col
        )

        if self.use_meta_learner:
            self.logger.info("   Fitting stacker on OOF quantile forecasts")

            oof_X = self.training_oof_quantile_forecast.drop(columns=[target_col])
            oof_y = self.training_oof_quantile_forecast[target_col]

            self._fit_meta_learner(oof_quantiles=oof_X, oof_target=oof_y)

        self.logger.info("Probabilistic Forecasting Model training complete.")


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate probabilistic and point forecasts.

        Returns a DataFrame with quantile columns (q_0.20 ... q_0.975)
        and a 'predictions' column (point forecast).
        """
        self.logger.info("Generating forecasts with ProbabilisticFramework")

        X = X.copy()
        X["DateTime"] = 0

        # stage-1: quantile forecasts
        quantile_preds = self.quantile_ensemble.predict(X)
        quantile_preds = self.apply_enterococci_constraints(quantile_preds)

        quantile_preds = self.apply_isotonic_monotonicity(quantile_preds)

        # stage-2: point forecast (stacker if available)
        if self.use_meta_learner and getattr(self, "meta_learner", None) is not None:
            point_forecast = pd.Series(
                self.meta_learner.predict(quantile_preds),
                index=quantile_preds.index,
                name="predictions",
            )
        else:
            point_forecast = quantile_preds.median(axis=1).rename("predictions")

        # assemble output & final constraints
        results = quantile_preds.copy()
        results["predictions"] = point_forecast
        results = self.apply_enterococci_constraints(results)

        return results

    def _fit_meta_learner(
        self,
        oof_quantiles: pd.DataFrame,
        oof_target:    pd.Series
    ) -> None:
        """
        Fit (or re-fit) the point-forecast stacker.

        Parameters
        ----------
        oof_quantiles : pd.DataFrame
            K quantile predictions from stage-1 models.
        oof_target    : pd.Series
            Ground-truth Enterococci counts for the same rows.
        """
        self.logger.info("      training LGBM stacker on %d rows",
                        len(oof_target))

        pf_cfg          = self.config["models"]["probabilistic_framework"]
        stack_params    = pf_cfg.get("meta_learner_params", {})
        exceed_cutoff   = pf_cfg.get("exceedance_cutoff", 280)
        early_rounds    = pf_cfg.get("meta_learner_early_stopping_rounds", 40)
        valid_frac      = pf_cfg.get("meta_learner_valid_fraction", 0.15)
        random_state    = pf_cfg.get("random_state", 42)

        default_params = dict(
            n_estimators      = 400,
            learning_rate     = 0.03,
            num_leaves        = 64,
            min_data_in_leaf  = 20,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            reg_lambda        = 1.0,
            boosting_type     = "gbdt",
            max_bin           = 255,
            random_state      = random_state,
            verbose           = -1,
        )
        default_params.update(stack_params)
        stack_params = default_params

        metric_name = f"recall_{exceed_cutoff}"
        def recall_exceed(y_true: np.ndarray, y_pred: np.ndarray):
            tp = np.logical_and(y_pred >= exceed_cutoff,
                                y_true >= exceed_cutoff).sum()
            fn = np.logical_and(y_pred <  exceed_cutoff,
                                y_true >= exceed_cutoff).sum()
            recall = tp / (tp + fn + 1e-12)
            return metric_name, recall, True

        X_tr, X_val, y_tr, y_val = train_test_split(
            oof_quantiles, oof_target,
            test_size   = valid_frac,
            random_state = random_state,
            shuffle     = True,
        )

        self.meta_learner = lgb.LGBMRegressor(**stack_params)

        self.meta_learner.fit(
            X_tr, y_tr,
            eval_set   = [(X_val, y_val)],
            eval_metric= recall_exceed,
            callbacks  = [lgb.early_stopping(early_rounds, verbose=False)],
        )

        best_iter = getattr(self.meta_learner, "best_iteration_", None)
        best_rec  = (self.meta_learner.best_score_
                    .get("valid_0", {})
                    .get(metric_name, np.nan))

        self.logger.info("      stacker fitted - best_iter=%s  %s=%.4f",
                        str(best_iter), metric_name, best_rec)

    def apply_enterococci_constraints(self, preds: pd.DataFrame | pd.Series):
        """ clip at 5 MPN/100 mL and round to int """
        clipped = np.maximum(preds, 5).round()
        return pd.DataFrame(clipped, index=preds.index,
                            columns=getattr(preds, "columns", None))

    def apply_isotonic_monotonicity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce monotonicity on quantile predictions using isotonic regression per row.
        """
        quantile_columns = sorted(df.columns, key=lambda x: float(x.split("_")[1]))
        tau = [float(q.split("_")[1]) for q in quantile_columns]

        df_sorted = df.copy()

        for idx, row in df.iterrows():
            values = row[quantile_columns].values

            ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            mono_vals = ir.fit_transform(tau, values)

            df_sorted.loc[idx, quantile_columns] = mono_vals

        return df_sorted

    def save(self, output_path: Path) -> None:
        self.logger.info(f"Saving Probabilistic Forecasting Model to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        self.logger.info("Model saved successfully.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProbabilisticForecastingModel":
        """
        Lower-level training method that accepts features and target separately.
        """
        self.logger.info("Fitting Probabilistic Forecasting Model using provided X and y.")

        self.feature_columns = X.columns.tolist()

        X = X.copy()
        X["DateTime"] = 0

        processed_data = pd.concat([X, y], axis=1)
        self.quantile_ensemble.train(processed_data)

        self.logger.info("Probabilistic Forecasting Model fit complete.")
        return self

    @classmethod
    def load(cls, input_path: Path) -> "ProbabilisticForecastingModel":
        model = joblib.load(input_path)
        return model
