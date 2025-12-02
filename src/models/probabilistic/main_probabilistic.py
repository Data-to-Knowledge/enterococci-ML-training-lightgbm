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
from sklearn.isotonic import IsotonicRegression

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Import the quantile ensemble model from our quantile modeling module
from src.models.probabilistic.quantile_modeling import ProbabilisticQuantileEnsembleModel
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CDF-based exceedance probability (matches inference repo logic)
# ══════════════════════════════════════════════════════════════════════════════
def prob_exceed_vectorised(Q: np.ndarray, p: np.ndarray, y_threshold: float, debug: bool = False) -> np.ndarray:
    """
    Vectorised piece-wise linear CDF inversion to compute P(Y > y_threshold)
    from a set of predicted quantiles.

    Parameters
    ----------
    Q : ndarray, shape (n_obs, n_q)
        Quantile predictions per row, ASCENDING in τ.
    p : ndarray, shape (n_q,)
        Quantile levels (0 < τ < 1), ASCENDING (e.g., 0.05, 0.2, ...).
    y_threshold : float
        The exceedance threshold (e.g. 280).
    debug : bool
        If True, prints small samples of intermediate arrays.

    Returns
    -------
    ndarray, shape (n_obs,)
        P(Y > y_threshold) for each observation.
    """
    Q = Q.astype(float, copy=False)
    n_obs, n_q = Q.shape

    # Locate segment k s.t. Q_k <= y < Q_{k+1}; clip to valid range
    idx = (Q < y_threshold).sum(axis=1) - 1
    idx = np.clip(idx, 0, n_q - 2)

    row = np.arange(n_obs)
    Q_lo = Q[row, idx]
    Q_hi = Q[row, idx + 1]
    p_lo = p[idx]
    p_hi = p[idx + 1]

    # Linear interpolation of CDF at y_threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        Fy = p_lo + (y_threshold - Q_lo) / (Q_hi - Q_lo) * (p_hi - p_lo)

    # Strict bounds: let equality be handled by interpolation
    below_min = y_threshold < Q[:, 0]
    above_max = y_threshold > Q[:, -1]
    Fy[below_min] = 0.0
    Fy[above_max] = 1.0

    # Degenerate segments (Q_hi == Q_lo) or NaNs from division → fall back to upper τ
    degenerate = (Q_hi == Q_lo) | np.isnan(Fy)
    Fy = np.where(degenerate, p_hi, Fy)

    if debug:
        print("idx sample:", idx[:5])
        print("Q_lo sample:", Q_lo[:5])
        print("Q_hi sample:", Q_hi[:5])
        print("p_lo sample:", p_lo[:5])
        print("p_hi sample:", p_hi[:5])
        print("Fy sample:", Fy[:5])
        print("P(Y>y) sample:", (1.0 - Fy)[:5])

    # P(Y > y) = 1 - F(y)
    return 1.0 - Fy


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
    # PREDICT (matches inference pipeline logic)
    # ─────────────────────────────────────────────────────────────
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
        alert_level = (point_pred > 280).astype(np.int8) * 2 + 1  # >280 → 3, else 1

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
        
    def apply_enterococci_constraints(self, preds: pd.DataFrame | pd.Series):
        """ clip at 5 MPN/100 mL and round to int """
        clipped = np.maximum(preds, 5).round()        # ← NumPy outputs ndarray
        return pd.DataFrame(clipped, index=preds.index,
                            columns=getattr(preds, "columns", None))

    
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