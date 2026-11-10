"""
Feature Importance Utilities

Generate feature importance plots and CSV reports.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureImportance:

    OUTPUT_DIR = Path("outputs/plots")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save(model, feature_names, title="Feature Importance"):

        # يتحقق إن الموديل يدعم Feature Importance
        if not hasattr(
            model.pipeline.named_steps["model"],
            "feature_importances_",
        ):
            logger.warning(
                "This model does not provide feature importance."
            )
            return

        importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": model.pipeline.named_steps["model"].feature_importances_,
            }
        )

        importance = (
            importance
            .sort_values(
                by="Importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        file_name = title.lower().replace(" ", "_")

        csv_path = FeatureImportance.OUTPUT_DIR / f"{file_name}.csv"

        importance.to_csv(
            csv_path,
            index=False,
        )

        # رسم أفضل 20 Feature
        top = importance.head(20)

        plt.figure(figsize=(10, 8))

        plt.barh(
            top["Feature"],
            top["Importance"],
        )

        plt.gca().invert_yaxis()

        plt.title(title)

        plt.xlabel("Importance")

        plt.tight_layout()

        image_path = (
        FeatureImportance.OUTPUT_DIR
        / f"{file_name}.png"
         )

        plt.savefig(
            image_path,
            dpi=300,
        )

        plt.close()

        logger.info(
            "Feature importance saved."
        )

        logger.info(
            "CSV : %s",
            csv_path,
        )

        logger.info(
            "Plot: %s",
            image_path,
        )