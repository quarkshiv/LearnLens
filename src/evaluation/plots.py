"""
Visualization Utilities
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


OUTPUT = Path("outputs/figures")

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


class EvaluationPlots:

    @staticmethod
    def confusion_matrix(
        model,
        X_test,
        y_test,
        filename="confusion_matrix.png",
    ):

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            ax=ax,
            cmap="Blues",
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT / filename,
            dpi=300,
        )

        plt.close()

    @staticmethod
    def feature_importance(
        model,
        feature_names,
        filename="feature_importance.png",
        top_n=20,
    ):

        estimator = getattr(
            model,
            "named_steps",
            {}
        ).get("model", model)

        if not hasattr(
            estimator,
            "feature_importances_",
        ):
            return

        importance = estimator.feature_importances_

        order = importance.argsort()[::-1][:top_n]

        plt.figure(
            figsize=(10, 6)
        )

        plt.barh(
            range(len(order)),
            importance[order][::-1],
        )

        plt.yticks(
            range(len(order)),
            [feature_names[i] for i in order[::-1]],
        )

        plt.xlabel("Importance")

        plt.tight_layout()

        plt.savefig(
            OUTPUT / filename,
            dpi=300,
        )

        plt.close()