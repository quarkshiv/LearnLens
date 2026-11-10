"""
Regression Evaluation Plots
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import PredictionErrorDisplay

OUTPUT = Path("outputs/plots")
OUTPUT.mkdir(parents=True, exist_ok=True)


class RegressionPlots:

    @staticmethod
    def prediction_vs_actual(
        model,
        X_test,
        y_test,
    ):

        y_pred = model.predict(X_test)

        plt.figure(figsize=(8, 8))

        plt.scatter(
            y_test,
            y_pred,
            alpha=0.5,
        )

        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())

        plt.plot(
            [mn, mx],
            [mn, mx],
            "r--",
        )

        plt.xlabel("Actual")

        plt.ylabel("Predicted")

        plt.title("Prediction vs Actual")

        plt.tight_layout()

        plt.savefig(
            OUTPUT / "prediction_vs_actual.png",
            dpi=300,
        )

        plt.close()

    @staticmethod
    def residual_plot(
        model,
        X_test,
        y_test,
    ):

        try:

            PredictionErrorDisplay.from_estimator(
                model.pipeline,
                X_test,
                y_test,
                kind="residual_vs_predicted",
            )

            plt.tight_layout()

            plt.savefig(
                OUTPUT / "residual_plot.png",
                dpi=300,
            )

            plt.close()

        except Exception:
            pass