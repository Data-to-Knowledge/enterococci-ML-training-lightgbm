# src/pipeline/main_pipeline.py
import argparse
import logging
from pathlib import Path
import yaml
import sys
import pandas as pd
import os
project_root = Path(__file__).resolve().parent.parent.parent  
sys.path.append(str(project_root))

from src.data.data_loader import DataLoader
from src.data.preprocessing import Preprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.model_factory import ModelFactory
from src.evaluation.time_series_evaluator import TimeSeriesEvaluator
from src.evaluation.evaluator import Evaluator
from src.utils.logging import setup_logger
# from src.visualization.forecast_plots import create_forecast_visualizations
from src.visualization.interactive_plots import create_interactive_forecast_plot
from src.visualization.dashboard import generate_dashboard
from src.config.paths import (
    TRAINING_DATA_PATH,
    MAIN_CONFIG_PATH
)

logger = setup_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Water Quality Forecasting Pipeline")
    parser.add_argument("--config", type=str, default="config/main_config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "predict", "all"], default="all", help="Pipeline execution mode")
    return parser.parse_args()

def run_pipeline(config_path: str, mode: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Starting pipeline in {mode} mode with config from {config_path}")

    data_loader = DataLoader(config_path)
    preprocessor = Preprocessor(config_path)
    feature_engineer = FeatureEngineer(config_path)
    model_factory = ModelFactory(config)

    data = data_loader.load_processed_data(TRAINING_DATA_PATH)
    data = preprocessor.clean_data(data)
    data["Enterococci"] = preprocessor.set_max_target_value(data["Enterococci"])
    data = feature_engineer.engineer_features(data)
    data = preprocessor.transform_catergorical_variable_type(data)

    logger.info(f"Processed data: {data.shape} rows, {data.columns.size} columns")

    model_name = "probabilistic_framework"
    trained_model = None
    test_forecast = {}

    if mode in ["train", "all"]:
        logger.info(f"Training {model_name}...")
        model = model_factory.get_model(model_name)
        model.train(data)
        trained_model = model
        output_path = Path("models") / f"{model_name}.joblib"
        model.save(output_path)
        logger.info(f"Model {model_name} saved to {output_path}")

    if mode in ["evaluate", "all"]:
        ts_evaluator = TimeSeriesEvaluator(config["evaluation"])
        logger.info(f"Evaluating {model_name}")
        
        # Run time series cross-validation
        results, test_pred, _ = ts_evaluator.evaluate_with_tscv(
            model_factory, data, model_name,
            config["data"].get("date_column", "DateTime"),
            config["data"].get("target_column", "Enterococci")
        )

        # ── 🔁 Run final inference on test_pred to get full quantile outputs ──
        try:
            logger.info("Generating predictions on test set for dashboard")
            model = model_factory.get_model(model_name)
            X_eval = test_pred.drop(columns=["Enterococci"])
            pred_df = model.predict(X_eval)

            # Combine base and predictions
            test_forecast[model_name] = pd.concat(
                [
                    test_pred[["DateTime", "SITE_NAME", "Enterococci"]],
                    pred_df
                ],
                axis=1
            )
        except Exception as e:
            logger.error(f"❌ Error generating dashboard predictions during evaluation: {e}")
            test_forecast[model_name] = test_pred  # fallback to base only

        # Save evaluation report
        report_path = Path(config["evaluation"].get("report_path", "reports/evaluation_results.json"))
        ts_evaluator.generate_report(report_path)
        logger.info(f"Evaluation report saved to {report_path}")


    if mode in ["predict", "all"]:
        pred_config = config.get("prediction", {})
        pred_data_path = Path(pred_config.get("data_path", config["data"]["prediction_data_path"]))
        logger.info(f"Loading prediction data from {pred_data_path}")
        pred_data = data_loader.load_processed_data(pred_data_path)
        pred_data = preprocessor.clean_data(pred_data)
        pred_data = feature_engineer.engineer_features(pred_data)
        pred_data["Enterococci"] = preprocessor.set_max_target_value(pred_data["Enterococci"])
        pred_data = preprocessor.transform_catergorical_variable_type(pred_data)

        target_column = config["data"].get("target_column", "Enterococci")
        date_column = config["data"].get("date_column", "DateTime")
        site_column = config["data"].get("site_column", "SITE_NAME")
        results_df = pred_data.copy()

        has_target = target_column in results_df.columns
        X_pred = pred_data.drop(columns=[target_column]) if has_target else pred_data
        y_true = pred_data[target_column] if has_target else None

        logger.info(f"Generating predictions for {model_name}")
        try:
            if trained_model is not None:
                model = trained_model
            else:
                from src.models.probabilistic.main_probabilistic import ProbabilisticForecastingModel
                model_path = Path(pred_config.get(f"{model_name}_model_path", f"models/{model_name}.joblib"))
                logger.info(f"Loading model from {model_path}")
                model = ProbabilisticForecastingModel.load(model_path)

            predictions = model.predict(X_pred)

            if isinstance(predictions, pd.DataFrame):
                # Keep predictions clean and separate — no prefixing
                test_forecast[model_name] = pd.concat(
                    [
                        results_df[["DateTime", "SITE_NAME", "Enterococci"]],
                        predictions
                    ],
                    axis=1
                )


            logger.info(f"Generated predictions for {model_name}")

        except Exception as e:
            logger.error(f"Error generating predictions for {model_name}: {e}")

        output_path = Path(pred_config.get("output_path", "results/predictions.csv"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        test_forecast[model_name].to_csv(output_path, index=False)
        logger.info(f"All predictions saved to {output_path}")

        if pred_config.get("generate_interactive_visualizations", True):
            interactive_path = Path(pred_config.get("interactive_visualization_path", "results/interactive"))
            interactive_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Generating interactive forecast visualizations")

            test_data = test_forecast[model_name].copy()    

            html_path = interactive_path / "forecast_comparison.html"
            create_interactive_forecast_plot(
                test_data=test_data,
                train_data=None,
                output_path=html_path,
                date_column=date_column,
                site_column=site_column,
                target_column=target_column
            )

            app = generate_dashboard(test_forecast, config)
            logger.info(f"Interactive visualization saved to {html_path}")

            return app



if __name__ == "__main__":
    args = parse_args()
    # run_pipeline(args.config, args.mode)
    app = run_pipeline(MAIN_CONFIG_PATH, "all")
    
    logger.info("Dashboard generated, starting server...")
    app.run(debug=False, port=8501)
