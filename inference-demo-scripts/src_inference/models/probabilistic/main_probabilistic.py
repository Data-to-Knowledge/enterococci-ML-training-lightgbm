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
from mapie.regression import ConformalizedQuantileRegressor
from sklearn.isotonic import IsotonicRegression

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Import the quantile ensemble model from our quantile modeling module
from src_inference.models.probabilistic.quantile_modeling import ProbabilisticQuantileEnsembleModel
from src_inference.utils.logging import setup_logger
from helpers.inference_data_preparation import *

logger = setup_logger(__name__)

class ProbabilisticForecastingModel:
    """
    Main model class for the Probabilistic Forecasting Framework.
    
    This model first trains an ensemble of LightGBM quantile regression models (stage 1).
    It then (optionally) uses a meta-learner to produce a point forecast and calibrates the prediction 
    intervals (stages 2 and 3). Currently, the meta-learner and calibration are placeholders; for 
    now, the point forecast is derived from the median quantile.
    
    This class implements the standard interface (train, predict, save, load) so that it integrates seamlessly
    with the main pipeline.
    """
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        # Data settings from configuration
        # self.target_column = config["data"].get("target_column", "Enterococci")
        # self.feature_columns = config["data"].get("feature_columns", None)

        self.target_column = None
        self.feature_columns = None
    
        # Initialize the quantile ensemble component using configuration settings
        self.quantile_ensemble = ProbabilisticQuantileEnsembleModel(config)
        
        # Placeholders for meta-learner and calibration components
        self.meta_learner = self.config["models"]["probabilistic_framework"].get("meta_learner")
        self.calibration_params = None

    def train(self, data: pd.DataFrame) -> None:
        """
        Train the probabilistic forecasting model.
        
        This includes:
          - Extracting features and target in the same manner as the LightGBM baseline.
          - Adding a dummy "DateTime" column as required.
          - Training the quantile ensemble.
        
        Args:
            data: Training data as a DataFrame.
        """
        self.logger.info("Starting training of Probabilistic Forecasting Model.")
        data.to_csv("data/processed/training_data.csv", index=False)

        # Extract features and target following the LightGBM benchmark pattern.
        X = data.drop('Enterococci', axis=1)
        y = data['Enterococci']
        X["DateTime"] = 0
        
        # Recombine features and target to form a processed DataFrame.
        self.training_data = pd.concat([X, y], axis=1)
        
        # Train the quantile ensemble on the processed data.
        self.training_oof_quantile_forecast = self.quantile_ensemble.train(self.training_data)

        self.logger.info("Probabilistic Forecasting Model training complete.")


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        1. raw quantile outputs  q_*_raw  (audit / SHAP alignment)
        2. isotonic regression   q_*      (monotone, τ-aligned)
        3. median(iso-adjusted)  → predictions
        4. CDF on mono grid      → prob_exceed_280
        5. traffic-light label   → risk_label
        6. alert level           → Enterococci Alert Level (1/3)
        7. timestamp             → time_UTC (top-of-hour)
        8. calendar date         → date (YYYY-MM-DD)
        """
        self.logger.info("Prob-forecast predict(): raw → mono → median.")

        # ── 0. Prep ────────────────────────────────────────────────
        X = X.copy()
        X["DateTime"] = 0

        # ── 1. RAW quantile ensemble outputs ───────────────────────
        q_raw = self.quantile_ensemble.predict(X)
        q_raw = self.apply_enterococci_constraints(q_raw)   # ≥5, int

        raw_cols = {c: f"{c}_raw" for c in q_raw.columns}
        q_raw_audit = q_raw.rename(columns=raw_cols)

        # ── 2. Cascading-max monotone fix ──────────────────────────
        q_mono = self.apply_isotonic_monotonicity(q_raw.copy())

        # ── 3. Point forecast (median of mono grid) ────────────────
        point_pred = q_mono.median(axis=1)

        # ── 4. Exceedance probability on mono grid ────────────────
        q_cols = q_mono.columns.tolist()
        taus   = np.array([float(c.split("_")[1]) for c in q_cols])
        prob_unsafe = prob_exceed_vectorised(
            q_mono[q_cols].to_numpy(float), taus, 280.0)

        # ── 5. Risk label ─────────────────────────────────────────
        risk = pd.cut(point_pred,
                    bins=[-np.inf, 280, np.inf],
                    labels=["SAFE", "EXCEED"],
                    right=True, include_lowest=True).astype(str)

        # ── 6. Enterococci Alert Level ────────────────────────────
        alert_level = (point_pred > 280).astype(np.int8) * 2 + 1  # >280 → 3, else 1, good programmatic math trick

        # ── 7. time_UTC ──────────────────────────────────────────
        time_utc = pd.Timestamp.now(tz="UTC").floor("h")

        # ── 8. date (derived from time_UTC) ──────────────────────
        date_val = time_utc.date()

        # ── 9. Assemble output ───────────────────────────────────
        out = pd.concat([q_mono, q_raw_audit], axis=1)
        out["predictions"]             = point_pred
        out["prob_exceed_280"]         = prob_unsafe
        out["risk_label"]              = risk
        out["Enterococci Alert Level"] = alert_level.astype("int64")
        out["time_UTC"]                = time_utc
        out["date"]                    = date_val

        out[q_cols + ["predictions"]] = self.apply_enterococci_constraints(
            out[q_cols + ["predictions"]])

        return out

# ────────────────────────────────────────────────────────────────────

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


# # ────────────────────────────────────────────────────────────────────
#     def predict(self, X: pd.DataFrame) -> pd.DataFrame:
#         """
#         Generate probabilistic predictions and risk classes.

#         Returns
#         -------
#         DataFrame with:
#         • q_*              : the 12 quantile predictions
#         • predictions      : point forecast (median, q_0.5)
#         • prob_exceed_280  : P(Enterococci > 280)
#         • risk_label       : SAFE | CAUTION | EXCEED  (0–140, 141–280, >280)
#         """
#         self.logger.info("Generating predictions using the Probabilistic Forecasting Model.")

#         # ─── 0. Prep ─────────────────────────────────────────────────
#         X = X.copy()
#         X["DateTime"] = 0                                # baseline requirement

#         # ─── 1. Quantile ensemble ──────────────────────────────────
#         quantile_preds = self.quantile_ensemble.predict(X)

#         # Constrain (≥5, integer) the quantile grid only
#         quantile_preds = self.apply_enterococci_constraints(quantile_preds)
#         self.training_oof_quantile_forecast = self.apply_enterococci_constraints(
#             self.training_oof_quantile_forecast
#         )

#         _ = self.calibrate_intervals(quantile_preds, X)     # keeps your CQR step
#         quantile_preds = self.monotonic_sort_quantiles(
#             quantile_preds, quantile_preds.columns
#         )

#         # ─── 2. Point forecast ─────────────────────────────────────
#         if self.meta_learner:
#             point_forecast = self.apply_meta_learner(
#                 self.training_oof_quantile_forecast, quantile_preds
#             )
#         else:
#             point_forecast = self.apply_average_quantile(quantile_preds)     # predictive median

#         # ─── 3. Exceedance probability (280-MPN guideline) ─────────
#         q_cols = [c for c in quantile_preds.columns if c.startswith("q_")]
#         taus   = np.array([float(c.split("_")[1]) for c in q_cols])
#         order  = np.argsort(taus)
#         taus   = taus[order]
#         q_cols = [q_cols[i] for i in order]

#         Q = quantile_preds[q_cols].to_numpy(float)
#         prob_unsafe = prob_exceed_vectorised(Q, taus, 280.0)
#         quantile_preds["prob_exceed_280"] = prob_unsafe

#         # ─── 4. Risk label from the point estimate ────────────────
#         quantile_preds["risk_label"] = pd.cut(
#             point_forecast,
#             bins=[-np.inf, 140, 280, np.inf],
#             labels=["SAFE", "CAUTION", "EXCEED"],
#             right=True,
#             include_lowest=True
#         )

#         # ─── 5. Assemble final output ─────────────────────────────
#         results = quantile_preds.copy()
#         results["predictions"] = point_forecast

#         # Final safety net on numeric concentration columns only
#         conc_cols = q_cols + ["predictions"]
#         results[conc_cols] = self.apply_enterococci_constraints(results[conc_cols])

#         return results
# ────────────────────────────────────────────────────────────────────

    def apply_meta_learner(self, train_quantile_preds: pd.DataFrame, test_quantile_preds: pd.DataFrame) -> pd.Series:
        """
        Combine the quantile predictions using a meta-learner to produce a point forecast.
        Future implementation: Use a gradient boosting model or another ensemble method to combine
        quantile predictions.
        
        Currently, as a placeholder, we use the median of the quantile predictions.
        """
        self.logger.info("Training meta-learner for point forecast.")

        # Prepare meta-learning data.
        X_meta = train_quantile_preds.drop('Enterococci', axis=1)
        y_meta = train_quantile_preds['Enterococci']

        # Check if tuning is enabled via config.
        if self.config.get("meta_learner_tune", False):
            self.logger.info("Performing grid search tuning for meta-learner.")
            param_grid = {
                "n_estimators": [100, 150, 180, 200, 250],
                "learning_rate": [0.01, 0.05, 0.087, 0.1, 0.15],
                "num_leaves": [20, 30, 40, 50, 60],
                "min_data_in_leaf": [3, 5, 10, 20],
                "colsample_bytree": [0.7, 0.8, 0.9, 0.958, 1.0],
                "reg_lambda": [0.1, 0.5, 0.654, 1.0, 2.0]
            }
            grid_search = GridSearchCV(
                estimator=lgb.LGBMRegressor(verbose=-1),
                param_grid=param_grid,
                cv=5,
                scoring="neg_mean_absolute_error",
                n_jobs=-1
            )
            grid_search.fit(X_meta, y_meta)
            best_params = grid_search.best_params_
            self.logger.info(f"Best meta-learner parameters: {best_params}")
            self.meta_learner = grid_search.best_estimator_
        else:
            # Use parameters from config or default values.
            meta_params = self.config.get("meta_learner_params", {
                "n_estimators": 180,
                "learning_rate": 0.08721057029770096,
                "num_leaves": 40,
                "min_data_in_leaf": 5,
                "colsample_bytree": 0.9587602766121369,
                "reg_lambda": 0.6540961177848345,
                "boosting_type": "gbdt"
            })
            self.logger.info(f"Using meta-learner parameters: {meta_params}")
            self.meta_learner = lgb.LGBMRegressor(**meta_params, verbose=-1)
            self.meta_learner.fit(X_meta, y_meta)
        
        # Predict using the meta-learner.
        meta_learn_preds = self.meta_learner.predict(test_quantile_preds)
        meta_learn_preds = pd.Series(meta_learn_preds, index=test_quantile_preds.index, name="predictions")
        
        # Apply enterococci constraints (ensuring domain-specific output bounds).
        meta_learn_preds = self.apply_enterococci_constraints(meta_learn_preds)
        
        return meta_learn_preds
    
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
        mapie = ConformalizedQuantileRegressor(
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
        
    def apply_enterococci_constraints(self, predictions):
        # Apply minimum value of 5 MPN/100mL
        constrained = np.maximum(predictions, 5)
        # Round to nearest integer
        constrained = np.round(constrained)
        return constrained
    

    ### the following is part of Asif's code
    ### this is the "global sort" monotone routine
    # def monotonic_sort_quantiles(self, df, quantile_columns):
    #     # row-wise loop
    #     for index, row in df.iterrows():
    #         # list-comprehension - pull out the 12 numeric predictions in the order given by quantile_columns
    #         # example result would be [5, 5, 6, 9, 15, 15, 15, 51, 11, 26, 20, 102]
    #         # sorted(...) – sorts that list ascending.
    #         # After sorting: [5, 5, 6, 9, 11, 15, 15, 15, 20, 26, 51, 102]
    #         # Guarantees monotonicity but loses the original τ (quantile) mapping.
    #         sorted_values = sorted([row[quantile] for quantile in quantile_columns])
            
    #         # write-back loop
    #         # walks through the column names in quantile_columns in their original order
    #         # writes the i-th smallest value into the i-th column
    #         for i, quantile in enumerate(quantile_columns):
    #             df.at[index, quantile] = sorted_values[i]

    #         # consequence: the value that lands in, say, q_0.85 is no longer the output of the τ=0.85 model; it is merely the 8-th order statistic of the 12 raw values.
    #         # anything downstream that relies on this method might be statistically incorrect
    #     return df
    

    ## the following is the cascading-max monotone routine
    ## this is a widely-recognised technique used by Amazon, energy companies, finance VaR models, etc. for predictive models that use quantile regression.
    # def monotonic_sort_quantiles(self, df, quantile_columns):
    #     """
    #     Enforces monotonicity in predicted quantiles (q_τ values stay aligned with their τs).
    #     Ensures q_τ ≤ q_τ+1 for all τ.
    #     """
    #     # loop over rows: loop over each sample (one site per hour)
    #     # idx is the DataFrame index, row is a Series holding that row's values
    #     for idx, row in df.iterrows():
            
    #         # Extract quantile vector, build an ordered Python list of the 12 raw predictions
    #         # e.g. [5, 5, 6, 9, 15, 15, 15, 51, 11, 26, 20, 102].
    #         values = [row[q] for q in quantile_columns]
            
    #         # start at the secodn element (index 1) and walk forward
    #         for i in range(1, len(values)):
    #             # force non-decreasing, so if a value is smaller than its left neighbour, replace it with the neighbour's value
    #             # so in effect, each step guarantees monotonicity, nothing is reorder, only lifted where needed
    #             values[i] = max(values[i], values[i - 1]) 

    #         # write-back, copy the corrected list back into the DataFrame column-by-column, preserving the original quantile label.
    #         # Example – the number in q_0.85 is still that model’s output (possibly lifted), never swapped with another τ.
    #         for j, q in enumerate(quantile_columns):
    #             df.at[idx, q] = values[j]
        
    #     # monotinicity is now guaranteed and τ-aligned
    #     return df

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