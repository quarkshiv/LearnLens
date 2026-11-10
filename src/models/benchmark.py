"""
Model Benchmark

Compare multiple machine learning models
using the same preprocessing pipeline.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .classification import ClassificationPipeline
from .regression import RegressionPipeline

from ..evaluation.metrics import EvaluationMetrics
from ..evaluation.report import ReportGenerator
from ..evaluation.feature_importance import FeatureImportance
from ..evaluation.classification_plots import ClassificationPlots
from ..evaluation.regression_plots import RegressionPlots
logger = logging.getLogger(__name__)

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)


class Benchmark:

    CLASSIFICATION_MODELS = [
        "logistic",
        "decision_tree",
        "random_forest",
    ]

    REGRESSION_MODELS = [
        "linear",
        "decision_tree",
        "random_forest",
    ]

    @staticmethod
    def classification(
        transformer,
        X_train,
        X_test,
        y_train,
        y_test,
    ):

        logger.info("=" * 60)
        logger.info("Classification Benchmark")
        logger.info("=" * 60)

        results = []

        best_model = None
        best_score = -1

        for model_name in Benchmark.CLASSIFICATION_MODELS:

            logger.info(f"Training {model_name}")

            start = time.perf_counter()

            model = ClassificationPipeline(
                transformer=transformer,
                model_name=model_name,
            )

            model.fit(
                X_train,
                y_train,
            )

            train_time = (
                time.perf_counter() - start
            )

            predictions = model.predict(
                X_test
            )

            metrics = EvaluationMetrics.classification(
                y_test,
                predictions,
            )

            metrics["Model"] = model_name

            metrics["Training Time"] = round(
                train_time,
                3,
            )

            results.append(metrics)

            if metrics["Accuracy"] > best_score:

                best_score = metrics["Accuracy"]

                best_model = model

        df = pd.DataFrame(results)

        df = df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "Training Time",
            ]
        ]

        ReportGenerator.save_metrics(
            {},
            "dummy.csv",
        )

        df.to_csv(
            OUTPUT / "classification_benchmark.csv",
            index=False,
        )

        logger.info(df)

        if best_model is not None:
            FeatureImportance.save(
             model=best_model,
             feature_names=best_model.pipeline.named_steps[
             "preprocessor"
             ].get_feature_names_out(),
             title="Classification Feature Importance",
            )
            ClassificationPlots.confusion_matrix(
                best_model,
                X_test,
                y_test,
            )
            ClassificationPlots.roc_curve(
                best_model,
                X_test,
                y_test,
            )
        return best_model, df

    @staticmethod
    def regression(
        transformer,
        X_train,
        X_test,
        y_train,
        y_test,
    ):

        logger.info("=" * 60)
        logger.info("Regression Benchmark")
        logger.info("=" * 60)

        results = []

        best_model = None
        best_score = float("-inf")

        for model_name in Benchmark.REGRESSION_MODELS:

            logger.info(f"Training {model_name}")

            start = time.perf_counter()

            model = RegressionPipeline(
                transformer=transformer,
                model_name=model_name,
            )

            model.fit(
                X_train,
                y_train,
            )

            train_time = (
                time.perf_counter() - start
            )

            predictions = model.predict(
                X_test
            )

            metrics = EvaluationMetrics.regression(
                y_test,
                predictions,
            )

            metrics["Model"] = model_name

            metrics["Training Time"] = round(
                train_time,
                3,
            )

            results.append(metrics)

            if metrics["R2"] > best_score:

                best_score = metrics["R2"]

                best_model = model

        df = pd.DataFrame(results)

        df = df[
            [
                "Model",
                "MAE",
                "MSE",
                "RMSE",
                "R2",
                "Training Time",
            ]
        ]

        df.to_csv(
            OUTPUT / "regression_benchmark.csv",
            index=False,
        )

        logger.info(df)

        if best_model is not None:
            FeatureImportance.save(
                model=best_model,
                feature_names=best_model.pipeline.named_steps[
               "preprocessor"
                ].get_feature_names_out(),
                title="Regression Feature Importance",
            )
            RegressionPlots.prediction_vs_actual(
                best_model,
                X_test,
                y_test,
            )

            RegressionPlots.residual_plot(
                best_model,
                X_test,
                y_test,
            )
        return best_model, df

    @staticmethod
    def run_all(
        transformer,
        X_train_cls,
        X_test_cls,
        y_train_cls,
        y_test_cls,
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg,
    ):

        logger.info("=" * 60)
        logger.info("Running Complete Benchmark")
        logger.info("=" * 60)

        best_cls_model, cls_results = (
            Benchmark.classification(
                transformer,
                X_train_cls,
                X_test_cls,
                y_train_cls,
                y_test_cls,
            )
        )

        best_reg_model, reg_results = (
            Benchmark.regression(
                transformer,
                X_train_reg,
                X_test_reg,
                y_train_reg,
                y_test_reg,
            )
        )

        logger.info("=" * 60)
        logger.info("Benchmark Completed")
        logger.info("=" * 60)

        return {
            "classification_model": best_cls_model,
            "classification_results": cls_results,
            "regression_model": best_reg_model,
            "regression_results": reg_results,
        }