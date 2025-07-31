# import logging
# from pathlib import Path
# from typing import Dict, Any
# import pandas as pd
# import numpy as np
# import joblib
# import sys
# import lightgbm as lgb
# from sklearn.model_selection import GridSearchCV
# import os
# from mapie.quantile_regression import MapieQuantileRegressor

# project_root = Path(__file__).resolve().parent.parent.parent.parent
# sys.path.append(str(project_root))

# # Import the quantile ensemble model from our quantile modeling module
# from src.models.probabilistic.quantile_modeling import ProbabilisticQuantileEnsembleModel
# from src.utils.logging import setup_logger

# logger = setup_logger(__name__)

# class ProbabilisticForecastingModel:
#     """
#     Main model class for the Probabilistic Forecasting Framework.
    
#     This model first trains an ensemble of LightGBM quantile regression models (stage 1).
#     It then (optionally) uses a meta-learner to produce a point forecast and calibrates the prediction 
#     intervals (stages 2 and 3). Currently, the meta-learner and calibration are placeholders; for 
#     now, the point forecast is derived from the median quantile.
    
#     This class implements the standard interface (train, predict, save, load) so that it integrates seamlessly
#     with the main pipeline.
#     """
#     def __init__(self, config: Dict[str, Any]):
#         self.logger = logging.getLogger(self.__class__.__name__)
#         self.config = config
        
#         # Data settings from configuration
#         # self.target_column = config["data"].get("target_column", "Enterococci")
#         # self.feature_columns = config["data"].get("feature_columns", None)

#         self.target_column = None
#         self.feature_columns = None
    
#         # Initialize the quantile ensemble component using configuration settings
#         self.quantile_ensemble = ProbabilisticQuantileEnsembleModel(config)
        
#         # Placeholders for meta-learner and calibration components
#         self.meta_learner = self.config["models"]["probabilistic_framework"].get("meta_learner")
#         self.calibration_params = None

#     def train(self, data: pd.DataFrame) -> None:
#         """
#         Train the probabilistic forecasting model.
        
#         This includes:
#           - Extracting features and target in the same manner as the LightGBM baseline.
#           - Adding a dummy "DateTime" column as required.
#           - Training the quantile ensemble.
        
#         Args:
#             data: Training data as a DataFrame.
#         """
#         self.logger.info("Starting training of Probabilistic Forecasting Model.")
#         data.to_csv("data/processed/training_data.csv", index=False)

#         # Extract features and target following the LightGBM benchmark pattern.
#         X = data.drop('Enterococci', axis=1)
#         y = data['Enterococci']
#         X["DateTime"] = 0
        
#         # Recombine features and target to form a processed DataFrame.
#         self.training_data = pd.concat([X, y], axis=1)
        
#         # Train the quantile ensemble on the processed data.
#         self.training_oof_quantile_forecast = self.quantile_ensemble.train(self.training_data)

#         self.logger.info("Probabilistic Forecasting Model training complete.")

#     def predict(self, X: pd.DataFrame) -> pd.DataFrame:
#         """
#         Generate predictions using the probabilistic forecasting model.
        
#         This method:
#           - Adds a dummy "DateTime" column as in the baseline.
#           - Obtains quantile predictions from the ensemble.
#           - Uses the median quantile as the point forecast.
        
#         Args:
#             X: Feature DataFrame for prediction.
        
#         Returns:
#             A DataFrame containing quantile predictions and a column 'point_forecast'.
#         """
#         self.logger.info("Generating predictions using the Probabilistic Forecasting Model.")
        
#         # Mimic the baseline by adding a "DateTime" column.
#         X = X.copy()  # Avoid modifying the original DataFrame.
#         X["DateTime"] = 0
        
#         # Obtain quantile predictions from the ensemble
#         quantile_preds = self.quantile_ensemble.predict(X)
        
#         # Apply enterococci constraints (e.g., non-negativity, upper bounds)
#         quantile_preds = self.apply_enterococci_constraints(quantile_preds)
#         self.training_oof_quantile_forecast = self.apply_enterococci_constraints(self.training_oof_quantile_forecast)
        
#         # Calibrate prediction intervals
#         lower_prediction_interval, upper_prediction_interval  = self.calibrate_intervals(quantile_preds, X)
#         # print(lower_prediction_interval)
#         # print(upper_prediction_interval)
        
#         # Generate point forecast using the meta-learner
#         if self.meta_learner:
#             point_forecast = self.apply_meta_learner(self.training_oof_quantile_forecast, quantile_preds)
#         else:
#             point_forecast = self.apply_average_quantile(quantile_preds)

#         quantile_preds = self.monotonic_sort_quantiles(quantile_preds, quantile_preds.columns)
        
#         # Append point forecast to quantile predictions
#         results = quantile_preds.copy()
#         results["predictions"] = point_forecast
        
#         # Ensure final results also satisfy enterococci constraints
#         results = self.apply_enterococci_constraints(results)
        
#         return results

    
#     def apply_meta_learner(self, train_quantile_preds: pd.DataFrame, test_quantile_preds: pd.DataFrame) -> pd.Series:
#         """
#         Combine the quantile predictions using a meta-learner to produce a point forecast.
#         Future implementation: Use a gradient boosting model or another ensemble method to combine
#         quantile predictions.
        
#         Currently, as a placeholder, we use the median of the quantile predictions.
#         """
#         self.logger.info("Training meta-learner for point forecast.")

#         # Prepare meta-learning data.
#         X_meta = train_quantile_preds.drop('Enterococci', axis=1)
#         y_meta = train_quantile_preds['Enterococci']

#         # Check if tuning is enabled via config.
#         if self.config.get("meta_learner_tune", False):
#             self.logger.info("Performing grid search tuning for meta-learner.")
#             param_grid = {
#                 "n_estimators": [100, 150, 180, 200, 250],
#                 "learning_rate": [0.01, 0.05, 0.087, 0.1, 0.15],
#                 "num_leaves": [20, 30, 40, 50, 60],
#                 "min_data_in_leaf": [3, 5, 10, 20],
#                 "colsample_bytree": [0.7, 0.8, 0.9, 0.958, 1.0],
#                 "reg_lambda": [0.1, 0.5, 0.654, 1.0, 2.0]
#             }
#             grid_search = GridSearchCV(
#                 estimator=lgb.LGBMRegressor(verbose=-1),
#                 param_grid=param_grid,
#                 cv=5,
#                 scoring="neg_mean_absolute_error",
#                 n_jobs=-1
#             )
#             grid_search.fit(X_meta, y_meta)
#             best_params = grid_search.best_params_
#             self.logger.info(f"Best meta-learner parameters: {best_params}")
#             self.meta_learner = grid_search.best_estimator_
#         else:
#             # Use parameters from config or default values.
#             meta_params = self.config.get("meta_learner_params", {
#                 "n_estimators": 180,
#                 "learning_rate": 0.08721057029770096,
#                 "num_leaves": 40,
#                 "min_data_in_leaf": 5,
#                 "colsample_bytree": 0.9587602766121369,
#                 "reg_lambda": 0.6540961177848345,
#                 "boosting_type": "gbdt"
#             })
#             self.logger.info(f"Using meta-learner parameters: {meta_params}")
#             self.meta_learner = lgb.LGBMRegressor(**meta_params, verbose=-1)
#             self.meta_learner.fit(X_meta, y_meta)
        
#         # Predict using the meta-learner.
#         meta_learn_preds = self.meta_learner.predict(test_quantile_preds)
#         meta_learn_preds = pd.Series(meta_learn_preds, index=test_quantile_preds.index, name="predictions")
        
#         # Apply enterococci constraints (ensuring domain-specific output bounds).
#         meta_learn_preds = self.apply_enterococci_constraints(meta_learn_preds)
        
#         return meta_learn_preds
    
#     def apply_average_quantile(self, quantile_preds: pd.DataFrame) -> pd.Series:
#         """
#         Combine the quantile predictions using a meta-learner to produce a point forecast.
#         Future implementation: Use a gradient boosting model or another ensemble method to combine
#         quantile predictions.
        
#         Currently, as a placeholder, we use the median of the quantile predictions.
#         """
#         self.logger.info("Applying meta-learner (placeholder): using median of quantile predictions.")
#         return quantile_preds.median(axis=1)

#     def calibrate_intervals(self, quantile_preds: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
#         """
#         Calibrate the prediction intervals to ensure proper coverage (e.g., 90% coverage).
#         Future implementation: Apply Conformalized Quantile Regression (CQR) or similar calibration.
        
#         Currently, as a placeholder, we return the quantile predictions unchanged.
#         """
#         self.logger.info("Calibrating prediction intervals")

#         base_model = self.quantile_ensemble.models[0.8]

#         X_train = self.training_data.drop('Enterococci', axis=1)
#         y_train = self.training_data['Enterococci']
        
#         # Create prediction intervals with symmetric alpha
#         mapie = MapieQuantileRegressor(
#             estimator=base_model,
#             method="quantile",
#             cv='split'
#         )
        
#         # Fit once
#         mapie.fit(X_train, y_train)

#         # Get predictions with proper confidence level
#         # alpha=0.1 for 90% prediction interval (0.05 on each side)
#         point_pred_test, intervals_test = mapie.predict(X_test, alpha=0.10)
#         point_pred_train, intervals_train = mapie.predict(X_train, alpha=0.10)
        
#         # Extract bounds
#         lower_pred_test = intervals_test[:, 0, 0]  # Lower bound
#         upper_pred_test = intervals_test[:, 1, 0]  # Upper bound
        
#         lower_pred_train = intervals_train[:, 0, 0]  # Lower bound
#         upper_pred_train = intervals_train[:, 1, 0]  # Upper bound


#         return lower_pred_test, upper_pred_test
        
#     def apply_enterococci_constraints(self, predictions):
#         # Apply minimum value of 5 MPN/100mL
#         constrained = np.maximum(predictions, 5)
#         # Round to nearest integer
#         constrained = np.round(constrained)
#         return constrained
    
#     def monotonic_sort_quantiles(self, df, quantile_columns):
#         for index, row in df.iterrows():
#             sorted_values = sorted([row[quantile] for quantile in quantile_columns])
#             for i, quantile in enumerate(quantile_columns):
#                 df.at[index, quantile] = sorted_values[i]
#         return df

#     def save(self, output_path: Path) -> None:
#         """
#         Save the probabilistic forecasting model to disk.
        
#         Args:
#             output_path: Path where the model should be saved.
#         """
#         self.logger.info(f"Saving Probabilistic Forecasting Model to {output_path}")
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         joblib.dump(self, output_path)
#         self.logger.info("Model saved successfully.")

#     def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProbabilisticForecastingModel":
#         """
#         Lower-level training method that accepts features and target separately.
        
#         Mimics the LightGBMModel.fit() behavior by adding a dummy "DateTime" column,
#         then recombining features and target before training the quantile ensemble.
        
#         Args:
#             X: Features DataFrame.
#             y: Target variable Series.
            
#         Returns:
#             Self reference for method chaining.
#         """
#         self.logger.info("Fitting Probabilistic Forecasting Model using provided X and y.")
        
#         # Save the feature names
#         self.feature_columns = X.columns.tolist()
        
#         # Mimic the baseline by adding a "DateTime" column.
#         X = X.copy()
#         X["DateTime"] = 0
        
#         # Recombine features and target to form a processed DataFrame.
#         processed_data = pd.concat([X, y], axis=1)
        
#         # Train the quantile ensemble on the processed data.
#         self.quantile_ensemble.train(processed_data)
        
#         self.logger.info("Probabilistic Forecasting Model fit complete.")
#         return self

#     @classmethod
#     def load(cls, input_path: Path) -> "ProbabilisticForecastingModel":
#         """
#         Load a previously saved probabilistic forecasting model.
        
#         Args:
#             input_path: Path to the saved model file.
            
#         Returns:
#             An instance of ProbabilisticForecastingModel.
#         """
#         model = joblib.load(input_path)
#         return model



import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import joblib
import sys
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import os
from mapie.quantile_regression import MapieQuantileRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import isotonic_regression, IsotonicRegression
from scipy.interpolate import PchipInterpolator


project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Import the quantile ensemble model from our quantile modeling module
from src.models.probabilistic.quantile_modeling import ProbabilisticQuantileEnsembleModel
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class ProbabilisticForecastingModel:
    """
    Probabilistic framework = 3-stage model
      1)  LightGBM quantile ensemble
      2)  (optional) point-forecast stacker (“meta-learner”)
      3)  (optional) interval calibration
    """
    def __init__(self, config: Dict[str, Any]):
        # ─── bookkeeping ──────────────────────────────────────────────────────
        self.logger  = logging.getLogger(self.__class__.__name__)
        self.config  = config                       # ← full YAML tree
        self.pf_cfg  = config["models"]["probabilistic_framework"]  # shortcut

        # ─── data / feature placeholders ─────────────────────────────────────
        self.target_column   = None
        self.feature_columns = None

        # ─── stage-1: quantile models ────────────────────────────────────────
        # initialise ensemble with ONLY the probabilistic sub-config
        self.quantile_ensemble = ProbabilisticQuantileEnsembleModel(self.pf_cfg)

        # ─── stage-2: meta-learner switch & object ───────────────────────────
        self.use_meta_learner = self.pf_cfg.get("meta_learner", False)  # bool flag
        self.meta_learner     = None        # will hold fitted lgb.LGBMRegressor

        # ─── stage-3: interval calibration placeholders ─────────────────────
        self.calibration_params = None      # kept for future conformal modules

    def train(self, data: pd.DataFrame) -> None:
        """
        Fit the full probabilistic-forecasting pipeline.

        Stage-1  →  one LightGBM per quantile
        Stage-2  →  optional LightGBM stacker (meta-learner)

        Parameters
        ----------
        data : pd.DataFrame
            Feature-engineered training set that already contains the target.
        """
        self.logger.info("▶︎ Training Probabilistic Forecasting Model …")

        # ───────────────────────── 1. Book-keeping ──────────────────────────
        target_col          = self.config["data"].get("target_column", "Enterococci")
        self.target_column  = target_col

        Path("data/processed").mkdir(parents=True, exist_ok=True)
        data.to_csv("data/processed/training_data.csv", index=False)          # reproducibility

        # ───────────────────────── 2. Add dummy DateTime ────────────────────
        # (keeps the feature signature identical to what the baseline models expect)
        train_df            = data.copy()
        train_df["DateTime"] = 0

        self.training_data  = train_df                                         # keep a copy

        # ───────────────────────── 3. Stage-1: quantile ensemble ────────────
        self.logger.info("   • Fitting LightGBM quantile ensemble …")
        self.training_oof_quantile_forecast = self.quantile_ensemble.train(
            train_df,
            target_col=target_col             # ← passes the dynamic name to the ensemble
        )

        # ───────────────────────── 4. Stage-2: stacker (optional) ───────────
        if self.use_meta_learner:
            self.logger.info("   • Fitting stacker on OOF quantile forecasts …")

            oof_X = self.training_oof_quantile_forecast.drop(columns=[target_col])
            oof_y = self.training_oof_quantile_forecast[target_col]

            self._fit_meta_learner(oof_quantiles=oof_X, oof_target=oof_y)

        self.logger.info("✔︎ Probabilistic Forecasting Model training complete.")


    # ─────────────────────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate probabilistic and point forecasts with risk evaluation.

        Steps
        -----
        1.  Add dummy “DateTime” column (LightGBM-compat).
        2.  Use the trained quantile ensemble to get the 12 quantile
            predictions for every row in *X*.
        3.  Apply isotonic regression to enforce monotonicity of quantiles.
        4.  Compute point forecast as median of isotonic quantiles.
        5.  Compute exceedance probability from CDF.
        6.  Assign risk label based on exceedance probability.
        7.  Return a DataFrame with:
                • q_0.20 … q_0.975       (12 columns)
                • predictions            (point forecast)
                • prob_exceed_280        (float)
                • risk_label             (str: SAFE or EXCEED)
        """
        self.logger.info("Generating forecasts with ProbabilisticFramework (training logic)")

        # 1️⃣ Dummy DateTime injection
        X = X.copy()
        X["DateTime"] = 0
        THRESHOLD_PROB = 0.18

        # 2️⃣ Stage-1: quantile forecasts (raw)
        q_raw = self.quantile_ensemble.predict(X)
        q_raw = self.apply_enterococci_constraints(q_raw)

        # 3️⃣ Stage-2: isotonic monotonicity fix
        q_iso = self.apply_isotonic_monotonicity(q_raw.copy())

        # 4️⃣ Point forecast: median of isotonic grid
        point_forecast = q_iso.median(axis=1).rename("predictions")

        # 5️⃣ Exceedance probability: P(y > 280) from monotonic CDF
        q_cols = q_iso.columns.tolist()  # assumes q_0.20 … q_0.975
        taus = np.array([float(c.split("_")[1]) for c in q_cols])

        prob_unsafe = ProbabilisticForecastingModel.prob_exceed_vectorised(q_iso[q_cols].to_numpy(float), taus, 280.0)

        # 6️⃣ Risk label from exceedance prob (threshold = 0.10 default logic)
        risk = np.where(prob_unsafe >= THRESHOLD_PROB, "EXCEED", "SAFE")

        # 7️⃣ Assemble final DataFrame
        results = q_iso.copy()
        results["predictions"] = point_forecast
        results["prob_exceed_280"] = prob_unsafe
        results["risk_label"] = risk

        # Final: clamp concentration predictions
        results[q_cols + ["predictions"]] = self.apply_enterococci_constraints(
            results[q_cols + ["predictions"]]
        )

        results.to_csv("training_forecasts.csv", index=False)


        return results


### using isotonic monotonicity from scikit learn (working version)

    # def predict(self, X: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Generate probabilistic and point forecasts.

    #     Steps
    #     -----
    #     1.  Add dummy “DateTime” column (LightGBM-compat).
    #     2.  Use the trained quantile ensemble to get the 12 quantile
    #         predictions for every row in *X*.
    #     3.  Apply isotonic regression to enforce monotonicity of quantiles.
    #     4.  Compute point forecast as median of isotonic quantiles.
    #     5.  Return a DataFrame with:
    #             • q_0.20 … q_0.975       (12 columns)
    #             • predictions            (point forecast)
    #     """
    #     self.logger.info("Generating forecasts with ProbabilisticFramework (training logic)")

    #     # 1️⃣ Dummy DateTime injection
    #     X = X.copy()
    #     X["DateTime"] = 0

    #     # 2️⃣ Stage-1: quantile forecasts (raw)
    #     q_raw = self.quantile_ensemble.predict(X)
    #     q_raw = self.apply_enterococci_constraints(q_raw)

    #     # 3️⃣ Stage-2: isotonic monotonicity fix
    #     q_iso = self.apply_isotonic_monotonicity(q_raw.copy())

    #     # 4️⃣ Point forecast: median of isotonic grid
    #     point_forecast = q_iso.median(axis=1).rename("predictions")

    #     # 5️⃣ Assemble final DataFrame
    #     results = q_iso.copy()
    #     results["predictions"] = point_forecast

    #     # Final: clamp concentration predictions
    #     q_cols = q_iso.columns.tolist()
    #     results[q_cols + ["predictions"]] = self.apply_enterococci_constraints(
    #         results[q_cols + ["predictions"]]
    #     )

    #     results.to_csv("training_forecasts.csv", index=False)

    #     return results


# #### monotone spline version with isotonic regression and then PCHIP spline through the plateau knots
#     def predict(self, X: pd.DataFrame) -> pd.DataFrame:
#         """
#         Generate probabilistic and point forecasts using the 2-step
#         PAVA → monotone-spline correction instead of plain isotonic.
#         """
#         self.logger.info("Generating forecasts with ProbabilisticFramework (mono-spline)")

#         # 1️⃣ Dummy DateTime
#         X = X.copy()
#         X["DateTime"] = 0

#         # 2️⃣ Raw quantile ensemble outputs
#         q_raw = self.quantile_ensemble.predict(X)
#         q_raw = self.apply_enterococci_constraints(q_raw)

#         # 3️⃣ Two-step monotone smoothing spline
#         q_smooth = self._apply_monotone_spline(q_raw)

#         # 4️⃣ Point forecast: median of smoothed grid
#         point_forecast = q_smooth.median(axis=1).rename("predictions")

#         # 5️⃣ Assemble & final clamp
#         results = q_smooth.copy()
#         results["predictions"] = point_forecast

#         cols = q_smooth.columns.tolist() + ["predictions"]
#         results[cols] = self.apply_enterococci_constraints(results[cols])

#         return results

    

    
# ---------------------------------------------------------------------------
# helper : fit the LightGBM stacker on OOF quantile predictions
# ---------------------------------------------------------------------------
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
        self.logger.info("      ↳ training LGBM stacker on %d rows …",
                        len(oof_target))

        # ── configuration --------------------------------------------------------
        pf_cfg          = self.config["models"]["probabilistic_framework"]
        stack_params    = pf_cfg.get("meta_learner_params", {})
        exceed_cutoff   = pf_cfg.get("exceedance_cutoff", 280)
        early_rounds    = pf_cfg.get("meta_learner_early_stopping_rounds", 40)
        valid_frac      = pf_cfg.get("meta_learner_valid_fraction", 0.15)
        random_state    = pf_cfg.get("random_state", 42)

        # sensible defaults (over-written by YAML, if present)
        default_params = dict(
            n_estimators      = 400,
            learning_rate     = 0.03,
            num_leaves        = 64,
            min_data_in_leaf  = 20,
            subsample         = 0.8,    # bagging_fraction
            colsample_bytree  = 0.8,
            reg_lambda        = 1.0,
            boosting_type     = "gbdt",
            max_bin           = 255,
            random_state      = random_state,
            verbose           = -1,
        )
        default_params.update(stack_params)
        stack_params = default_params

        # ── custom metric : recall on exceedance rows ---------------------------
        metric_name = f"recall_≥{exceed_cutoff}"
        def recall_exceed(y_true: np.ndarray, y_pred: np.ndarray):
            tp = np.logical_and(y_pred >= exceed_cutoff,
                                y_true >= exceed_cutoff).sum()
            fn = np.logical_and(y_pred <  exceed_cutoff,
                                y_true >= exceed_cutoff).sum()
            recall = tp / (tp + fn + 1e-12)
            # return (name, value, higher_is_better)
            return metric_name, recall, True

        # ── simple hold-out split for early-stopping ----------------------------
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            oof_quantiles, oof_target,
            test_size   = valid_frac,
            random_state = random_state,
            shuffle     = True,
        )

        # ── fit the stacker -----------------------------------------------------
        self.meta_learner = lgb.LGBMRegressor(**stack_params)

        self.meta_learner.fit(
            X_tr, y_tr,
            eval_set   = [(X_val, y_val)],
            eval_metric= recall_exceed,
            callbacks  = [lgb.early_stopping(early_rounds, verbose=False)],
        )

        # ── log best iteration & metric ----------------------------------------
        best_iter = getattr(self.meta_learner, "best_iteration_", None)
        # best_score_ looks like: {'training': {'recall_≥280': …},
        #                          'valid_0' : {'recall_≥280': …}}
        best_rec  = (self.meta_learner.best_score_
                    .get("valid_0", {})
                    .get(metric_name, np.nan))

        self.logger.info("      ✓ stacker fitted – best_iter=%s  %s=%.4f",
                        str(best_iter), metric_name, best_rec)


    
    
    def apply_average_quantile(self, quantile_preds: pd.DataFrame) -> pd.Series:
        """
        Combine the quantile predictions using a meta-learner to produce a point forecast.
        Future implementation: Use a gradient boosting model or another ensemble method to combine
        quantile predictions.
        
        Currently, as a placeholder, we use the median of the quantile predictions.
        """
        self.logger.info("Applying meta-learner (placeholder): using median of quantile predictions.")
        return quantile_preds.median(axis=1)

    def calibrate_intervals(self, quantile_preds: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Calibrate the prediction intervals to ensure proper coverage (e.g., 90% coverage).
        Future implementation: Apply Conformalized Quantile Regression (CQR) or similar calibration.
        
        Currently, as a placeholder, we return the quantile predictions unchanged.
        """
        self.logger.info("Calibrating prediction intervals")

        base_model = self.quantile_ensemble.models[0.8]

        X_train = self.training_data.drop('Enterococci', axis=1)
        y_train = self.training_data['Enterococci']
        
        # Create prediction intervals with symmetric alpha
        mapie = MapieQuantileRegressor(
            estimator=base_model,
            method="quantile",
            cv='split'
        )
        
        # Fit once
        mapie.fit(X_train, y_train)

        # Get predictions with proper confidence level
        # alpha=0.1 for 90% prediction interval (0.05 on each side)
        point_pred_test, intervals_test = mapie.predict(X_test, alpha=0.10)
        point_pred_train, intervals_train = mapie.predict(X_train, alpha=0.10)
        
        # Extract bounds
        lower_pred_test = intervals_test[:, 0, 0]  # Lower bound
        upper_pred_test = intervals_test[:, 1, 0]  # Upper bound
        
        lower_pred_train = intervals_train[:, 0, 0]  # Lower bound
        upper_pred_train = intervals_train[:, 1, 0]  # Upper bound


        return lower_pred_test, upper_pred_test
    
    @staticmethod
    def prob_exceed_vectorised(Q: np.ndarray, p: np.ndarray, y_threshold: float, debug=False) -> np.ndarray:
        """
        Vectorised piece-wise linear CDF inversion.
        Parameters
        ----------
        Q : ndarray, shape (n_obs, n_q)
            Quantile predictions per row, ASCENDING in τ. (one site-hour per observation)
        p : ndarray, shape (n_q,)
            Quantile levels (0 < τ < 1), ASCENDING. (like 0.05, 0.2, 0.35, 0.5, 0.6 etc.)
        y_threshold : float
            The exceedance threshold (e.g. 280).
        Returns
        -------
        ndarray, shape (n_obs,)
            P(Y > y_threshold) for each observation.
        """
        Q = Q.astype(float, copy=False)          # ensure float maths
        
        # n_obs is the number of predictions and n_q is the number of quantiles (12)
        n_obs, n_q = Q.shape

        ## --- 1. Locate the interval index k s.t. Q_k <= y < Q_{k+1} ----------
        ## For each row (site-hour), this finds which interval of the predicted quantiles the threshold (280) sits in
        ## example: If Q = [50, 120, 200, 310, 400] then the threshold of 280 lies between 200 and 310, which means it is at index 2

        idx = (Q < y_threshold).sum(axis=1) - 1       # -1 so that y==Q0 → idx=-1, y==Q1 → idx=0
        idx = np.clip(idx, 0, n_q - 2)                # keep in range [0, n_q-2], otherwise will be out of bounds

        if debug:
            print("idx (interval index):", idx[:5])

        # --- 2. Gather bounding quantiles and τ levels ----------------------
        ## This grabs the two quantile predictions that bracket the threshold (280), and the corresponding quantile levels (p_lo and p_hi)
        ## We can then interpolate between these to get the probability of exceeding the threshold
        row = np.arange(n_obs)
        Q_lo = Q[row, idx]
        Q_hi = Q[row, idx + 1]
        p_lo = p[idx]
        p_hi = p[idx + 1]

        if debug:
            print("Q_lo  sample:", Q_lo[:5])
            print("Q_hi  sample:", Q_hi[:5])
            print("p_lo  sample:", p_lo[:5])
            print("p_hi  sample:", p_hi[:5])

        # --- 3. Linear interpolation of F(y) --------------------------------
        ## We estimate the value of the cumulative distribution function at the threshold (280) by drawing a straight line between the two quantile points
        with np.errstate(divide="ignore", invalid="ignore"):
            Fy = p_lo + (y_threshold - Q_lo) / (Q_hi - Q_lo) * (p_hi - p_lo)

        # --- 4. Handle y below min or above max -----------------------------
        
        ## If the threshold is below the lowest quantile, then P(Y ≤ threshold) = 0, if above the highest quantile, then P(Y ≤ threshold) = 1
        ## and if interpolation caused a NaN (e.g., dividing by zero), it fills it with the upper quantile level

        Fy[y_threshold <= Q[:, 0]] = 0.0   # below or equal to lowest quantile
        Fy[y_threshold >= Q[:, -1]] = 1.0  # above or equal to highest quantile
        Fy = np.nan_to_num(Fy, nan=p_hi)   # in case Q_hi == Q_lo

        # debug statements
        if debug:
            print("Fy sample:", Fy[:5])
            print("prob_exceed sample:", (1.0 - Fy)[:5])

        # Fy is the CDF at the threshiold, i.e., P(Y ≤ 280)
        # so 1.0 - Fy gives us P(Y > 280) = exceedance probability
        return 1.0 - Fy     
        
    def apply_enterococci_constraints(self, preds: pd.DataFrame | pd.Series):
        """ clip at 5 MPN/100 mL and round to int """
        clipped = np.maximum(preds, 5).round()        # ← NumPy outputs ndarray
        return pd.DataFrame(clipped, index=preds.index,
                            columns=getattr(preds, "columns", None))
    
    # ─────────────────────────────────────────────────────────────
    #  NEW helper: row-wise monotone smoothing spline across τ
    # ─────────────────────────────────────────────────────────────
    def _apply_monotone_spline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Two-step post-processing for each row:
          1) Isotonic regression (PAVA)  → step-wise monotone grid
          2) PCHIP spline through plateau knots → smooth, still monotone
        Returns DataFrame with the same columns, now C¹-smooth & ordered.
        """
        q_cols = sorted(df.columns, key=lambda c: float(c.split("_")[1]))
        tau    = np.array([float(c.split("_")[1]) for c in q_cols])
        out_df = df.copy()

        def smooth_row(v: np.ndarray) -> np.ndarray:
            # 1️⃣ PAVA
            iso = isotonic_regression(v, increasing=True)
            iso = np.asarray(iso, dtype=float)

            # 2️⃣ find first index of each plateau
            jumps   = np.diff(iso, prepend=iso[0]-1)
            knot_ix = np.nonzero(jumps)[0]
            if knot_ix.size < 2:                 # all equal
                return iso

            spline = PchipInterpolator(tau[knot_ix], iso[knot_ix], extrapolate=True)
            return spline(tau)

        out_df[q_cols] = np.apply_along_axis(smooth_row, 1, df[q_cols].to_numpy(float))
        return out_df


    
    def monotonic_sort_quantiles(self, df, quantile_columns):
        for index, row in df.iterrows():
            sorted_values = sorted([row[quantile] for quantile in quantile_columns])
            for i, quantile in enumerate(quantile_columns):
                df.at[index, quantile] = sorted_values[i]
        return df
    
    from sklearn.isotonic import IsotonicRegression

    def apply_isotonic_monotonicity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce monotonicity on quantile predictions using isotonic regression per row.
        
        Args:
            df: DataFrame with quantile predictions as columns (e.g., q_0.05, q_0.1, ..., q_0.975)
            
        Returns:
            A new DataFrame with monotonic quantile predictions per row.
        """
        # Extract quantile columns and sort by tau
        quantile_columns = sorted(df.columns, key=lambda x: float(x.split("_")[1]))
        tau = [float(q.split("_")[1]) for q in quantile_columns]

        df_sorted = df.copy()

        for idx, row in df.iterrows():
            values = row[quantile_columns].values

            # Fit isotonic regressor on quantile levels (tau) → predicted values
            ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            mono_vals = ir.fit_transform(tau, values)

            df_sorted.loc[idx, quantile_columns] = mono_vals

        return df_sorted


    def save(self, output_path: Path) -> None:
        """
        Save the probabilistic forecasting model to disk.
        
        Args:
            output_path: Path where the model should be saved.
        """
        self.logger.info(f"Saving Probabilistic Forecasting Model to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        self.logger.info("Model saved successfully.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProbabilisticForecastingModel":
        """
        Lower-level training method that accepts features and target separately.
        
        Mimics the LightGBMModel.fit() behavior by adding a dummy "DateTime" column,
        then recombining features and target before training the quantile ensemble.
        
        Args:
            X: Features DataFrame.
            y: Target variable Series.
            
        Returns:
            Self reference for method chaining.
        """
        self.logger.info("Fitting Probabilistic Forecasting Model using provided X and y.")
        
        # Save the feature names
        self.feature_columns = X.columns.tolist()
        
        # Mimic the baseline by adding a "DateTime" column.
        X = X.copy()
        X["DateTime"] = 0
        
        # Recombine features and target to form a processed DataFrame.
        processed_data = pd.concat([X, y], axis=1)
        
        # Train the quantile ensemble on the processed data.
        self.quantile_ensemble.train(processed_data)
        
        self.logger.info("Probabilistic Forecasting Model fit complete.")
        return self

    @classmethod
    def load(cls, input_path: Path) -> "ProbabilisticForecastingModel":
        """
        Load a previously saved probabilistic forecasting model.
        
        Args:
            input_path: Path to the saved model file.
            
        Returns:
            An instance of ProbabilisticForecastingModel.
        """
        model = joblib.load(input_path)
        return model