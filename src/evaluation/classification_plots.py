"""
Classification Evaluation Plots
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

OUTPUT = Path("outputs/plots")
OUTPUT.mkdir(parents=True, exist_ok=True)


class ClassificationPlots:

    @staticmethod
    def confusion_matrix(
        model,
        X_test,
        y_test,
    ):

        predictions = model.predict(X_test)

        cm = confusion_matrix(
            y_test,
            predictions,
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
        )

        disp.plot()

        plt.tight_layout()

        plt.savefig(
            OUTPUT / "confusion_matrix.png",
            dpi=300,
        )

        plt.close()

    @staticmethod
    def roc_curve(
        model,
        X_test,
        y_test,
    ):

        if not hasattr(
            model,
            "predict_proba",
        ):
            return

        try:

            RocCurveDisplay.from_estimator(
              model.pipeline,
                X_test,
                y_test,
            )

            plt.tight_layout()

            plt.savefig(
                OUTPUT / "roc_curve.png",
                dpi=300,
            )

            plt.close()

        except Exception:

            # Skip multiclass ROC
            pass