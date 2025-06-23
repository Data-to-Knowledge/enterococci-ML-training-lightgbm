from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from termcolor import colored
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class EvaluatorThreeClass:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = config.get("metrics", ["rmse", "mae", "r2", "mape", "nrmse"])
        self.exceedance_threshold = config.get("exceedance_threshold", 280)
        self.precautionary_threshold = config.get("precautionary_threshold", 140)
        self.results = {}

    def convert_target_to_flag(self, y):
        return pd.Series(
            ["SAFE" if val < self.precautionary_threshold
             else "CAUTION" if val < self.exceedance_threshold
             else "EXCEEDANCE" for val in y]
        )

    def evaluate_model(self, model: Any, model_name: str, X: pd.DataFrame, y: pd.Series,
                       fold_name: str = "fold_unspecified", dataset_name: str = "test") -> Tuple[Dict[str, float], pd.DataFrame]:
        y_pred_fold = model.predict(X)
        if model_name == "probabilistic_framework":
            y_pred = y_pred_fold['predictions']
        else:
            y_pred = y_pred_fold.copy()

        y_pred.index = y.index
        metrics_results = self._calculate_metrics(y, y_pred)

        if fold_name not in self.results:
            self.results[fold_name] = {}
        self.results[fold_name] = metrics_results

        return metrics_results, y_pred_fold

    def _calculate_metrics(self, y_true: pd.Series, y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        results = {}
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        if "rmse" in self.metrics:
            results["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        if "mae" in self.metrics:
            results["mae"] = mean_absolute_error(y_true, y_pred)
        if "r2" in self.metrics:
            results["r2"] = r2_score(y_true, y_pred)
        if "mape" in self.metrics:
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
            results["mape"] = mape
        if "nrmse" in self.metrics:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            nrmse = rmse / (np.max(y_true) - np.min(y_true))
            results["nrmse"] = nrmse

        # 3-class classification metrics
        y_true_cls = self.convert_target_to_flag(y_true)
        y_pred_cls = self.convert_target_to_flag(y_pred)

        report = classification_report(y_true_cls, y_pred_cls, output_dict=True, zero_division=0)
        for label in ["SAFE", "CAUTION", "EXCEEDANCE"]:
            for metric in ["precision", "recall", "f1-score"]:
                key = f"{label.lower()}_{metric}"
                results[key] = round(report.get(label, {}).get(metric, 0.0), 4)

        # overall accuracy and macro avg
        results["accuracy"] = round(report.get("accuracy", 0.0), 4)
        for metric in ["precision", "recall", "f1-score"]:
            results[f"macro_{metric}"] = round(report.get("macro avg", {}).get(metric, 0.0), 4)

        return results

    def performance_evaluation_old_pipeline(self, train_forecast, test_forecast, model):
        y_pred_train = train_forecast["predictions"]
        y_pred_test = test_forecast["predictions"]
        y_train = train_forecast["Enterococci"]
        y_test = test_forecast["Enterococci"]

        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)

        print(colored("\n=== TRAIN METRICS ===", "green"))
        for k, v in train_metrics.items():
            print(f"{k}: {v:.4f}")
        print(colored("\n=== TEST METRICS ===", "blue"))
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")


    def generate_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        if not self.results:
            logger.warning("No evaluation results to report")
            return {}

        raw_report = {
            "results": self.results,
            "exceedance_threshold": self.exceedance_threshold,
            "precautionary_threshold": self.precautionary_threshold
        }

        # Optional: pretty-print the report for debug
        import pprint; pprint.pprint(raw_report, depth=4)

        def fallback_converter(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            return str(obj)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(raw_report, f, indent=4, default=fallback_converter)
            logger.info(f"Evaluation report saved to {output_path}")

        return raw_report





