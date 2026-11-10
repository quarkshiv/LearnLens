"""
Evaluation Metrics
"""

from __future__ import annotations

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class EvaluationMetrics:

    @staticmethod
    def classification(y_true, y_pred):

        metrics = {

            "Accuracy": accuracy_score(
                y_true,
                y_pred,
            ),

            "Precision": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "Recall": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "F1 Score": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

        }

        return metrics

    @staticmethod
    def regression(y_true, y_pred):

        metrics = {

            "MAE": mean_absolute_error(
                y_true,
                y_pred,
            ),

            "MSE": mean_squared_error(
                y_true,
                y_pred,
            ),

            "RMSE": mean_squared_error(
                y_true,
                y_pred,
            ) ** 0.5,

            "R2": r2_score(
                y_true,
                y_pred,
            ),

        }

        return metrics

    @staticmethod
    def confusion(y_true, y_pred):

        return confusion_matrix(
            y_true,
            y_pred,
        )

    @staticmethod
    def classification_table(
        y_true,
        y_pred,
    ):

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        return pd.DataFrame(report).T