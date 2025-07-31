from pathlib import Path
from typing import Dict, Any
import pandas as pd

from src.evaluation.evaluator import Evaluator
from src.evaluation.cross_validation import TimeSeriesCV
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

class TimeSeriesEvaluator(Evaluator):
    """Evaluator for time-series cross-validation.

    This class extends the base Evaluator to implement time-series specific
    evaluation strategies.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the time-series evaluator.

        Args:
            config: Evaluation configuration dictionary
        """
        super().__init__(config)
        self.tscv = TimeSeriesCV(config.get("cross_validation", {}))

    def evaluate_with_tscv(self, model_factory, data: pd.DataFrame, 
                         model_name: str, date_column: str = "sampling_date",
                         target_column: str = "enterococci") -> Dict[str, Any]:
        """Evaluate a model using time-series cross-validation.

        Args:
            model_factory: Factory to create model instances
            data: DataFrame containing features and target
            model_name: Name of the model to evaluate
            date_column: Name of the column containing dates
            target_column: Name of the column containing the target variable

        Returns:
            Dictionary containing evaluation results for each fold
        """
        # Get train-test splits
        splits = self.tscv.split(data, date_column)

        test_fold_results = {}
        train_fold_results = {}
        test_predictions = pd.DataFrame()
        train_predictions = pd.DataFrame()

        for i, (train_data, test_data) in enumerate(splits):
            fold_name = f"fold_{i+1}"
            logger.info(f"Evaluating {model_name} on {fold_name}")

            # Split features and target
            X_train = train_data.drop(columns=[target_column]).reset_index(drop=True)
            y_train = train_data[target_column].reset_index(drop=True)

            X_test = test_data.drop(columns=[target_column]).reset_index(drop=True)
            y_test = test_data[target_column].reset_index(drop=True)
            datetime_test, datetime_train = X_test["DateTime"], X_train["DateTime"]

            # Create and train the model
            model = model_factory.get_model(model_name)
            model.train(pd.concat([X_train, y_train], axis=1))

            # Evaluate the model
            test_fold_results[fold_name], y_pred_test = self.evaluate_model(
                model, model_name, X_test, y_test, dataset_name="test", fold_name=fold_name)

            train_fold_results[fold_name], y_pred_train = self.evaluate_model(
                model, model_name, X_train, y_train, dataset_name="train", fold_name=fold_name)

            y_pred_test = pd.DataFrame(y_pred_test)
            y_pred_test["DateTime"] = datetime_test
            y_pred_test["SITE_NAME"] = X_test["SITE_NAME"]
            y_pred_test["Enterococci"] = y_test

            y_pred_train = pd.DataFrame(y_pred_train)
            y_pred_train["DateTime"] = datetime_train
            y_pred_train["SITE_NAME"] = X_train["SITE_NAME"]
            y_pred_train["Enterococci"] = y_train

            test_predictions = pd.concat([test_predictions, y_pred_test], axis=0)
            train_predictions = pd.concat([train_predictions, y_pred_train], axis=0)

        train_predictions.reset_index(drop=True, inplace=True)
        test_predictions.reset_index(drop=True, inplace=True)

        self.test_predictions = test_predictions
        self.train_predictions = train_predictions

        # Use classification evaluation for probabilistic model only
        if model_name == "probabilistic_framework":
            logger.info("Running classification evaluation for risk labels...")
            y_true_labels = test_predictions["Enterococci"]
            y_pred_labels = test_predictions["risk_label"]

            classification_results = self.evaluate_probabilistic_risk_labels(y_true_labels, y_pred_labels)
            test_fold_results["overall"] = classification_results

            for metric, value in classification_results.items():
                logger.info(f"{model_name} {metric}: {value:.4f}")
        else:
            logger.warning(f"Skipping evaluation for model {model_name} (non-probabilistic)")
            test_fold_results["overall"] = {}

        self.results = {
            'fold_results': test_fold_results,
            'fold_predictions': test_predictions
        }

        return self.results, self.test_predictions, self.train_predictions
