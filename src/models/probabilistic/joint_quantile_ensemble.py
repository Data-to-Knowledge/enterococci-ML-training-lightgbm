import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Dict, Union
from functools import partial
from itertools import repeat, chain
from typing import Any, Dict



def _grad_rho(u: np.ndarray, alpha: float) -> np.ndarray:
    """Gradient of pinball loss."""
    return -(alpha - (u < 0).astype(float))


def composite_quantile_loss(
    y_pred: np.ndarray,
    dtrain: lgb.Dataset,
    alphas: List[float],
) -> (np.ndarray, np.ndarray):
    """
    Custom objective function for joint quantile regression.
    Computes gradient and (dummy) hessian for LightGBM.
    """
    n_alphas = len(alphas)
    y_true = dtrain.get_label()

    y_pred = y_pred.reshape(n_alphas, -1)
    y_true = y_true.reshape(n_alphas, -1)

    grads = []
    for i in range(n_alphas):
        err = y_true[i] - y_pred[i]
        grad = _grad_rho(err, alphas[i])
        grads.append(grad)

    grad = np.concatenate(grads)
    hess = np.ones_like(grad)  # dummy hessian
    return grad, hess


def _validate_alphas(alphas: Union[float, List[float]]) -> List[float]:
    if isinstance(alphas, float):
        return [alphas]
    return sorted(alphas)


def _prepare_features(X: Union[pd.DataFrame, pd.Series, np.ndarray], alphas: List[float]) -> pd.DataFrame:
    if isinstance(X, (np.ndarray, pd.Series)):
        X = pd.DataFrame(X)
    assert "_tau" not in X.columns, "Column name '_tau' is reserved."
    repeated_X = pd.concat([X] * len(alphas), ignore_index=True)
    tau_column = list(chain.from_iterable(repeat(alpha, len(X)) for alpha in alphas))
    repeated_X["_tau"] = tau_column
    return repeated_X


def _prepare_training_data(X, y, alphas):
    X_repeat = _prepare_features(X, alphas)
    y_repeat = np.concatenate([y] * len(alphas))
    return X_repeat, y_repeat


class JointQuantileEnsemble:
    def __init__(
        self,
        alphas: Union[List[float], float],
        params: Dict,
    ):
        self.alphas = _validate_alphas(alphas)
        self.params = params.copy()
        self.model = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        X_train, y_train = _prepare_training_data(X, y, self.alphas)
        dtrain = lgb.Dataset(data=X_train, label=y_train)

        # Add monotonic constraint on _tau
        constraints = [1 if col == '_tau' else 0 for col in X_train.columns]
        self.params["monotone_constraints"] = constraints

        fobj = partial(composite_quantile_loss, alphas=self.alphas)

        self.model = lgb.train(
            params=self.params,
            train_set=dtrain,
            fobj=fobj,
            verbose_eval=False
        )

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_aug = _prepare_features(X, self.alphas)
        preds = self.model.predict(X_aug)
        preds = preds.reshape(len(self.alphas), -1).T

        # Return as DataFrame with q_XX column names
        pred_df = pd.DataFrame(preds, columns=[f"q_{str(alpha)}" for alpha in self.alphas])
        return pred_df

    def save(self, path: str):
        if self.model:
            self.model.save_model(path)

    def load(self, path: str):
        self.model = lgb.Booster(model_file=path)


class JointProbabilisticForecastingModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quantiles = config["quantiles"]
        self.lgb_params = config["lgb_params"]
        self.calibration = config.get("calibration", False)


    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.train(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = self.model.predict(X)

        # Add point forecast: median of 12 quantiles (not q_0.5 specifically)
        preds["predictions"] = preds.median(axis=1)

        return preds

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model.load(path)

