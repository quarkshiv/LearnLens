"""
Inference Pipeline
"""

from __future__ import annotations

import logging

import pandas as pd

from src.utils.model_io import ModelIO


logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Run inference using previously trained models.
    """

    def __init__(self):

        self.classification_model = (
            ModelIO.load_classification_model()
        )

        self.regression_model = (
            ModelIO.load_regression_model()
        )

    def predict_classification(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Running classification inference..."
        )

        predictions = (
            self.classification_model.predict(X)
        )

        return pd.Series(
            predictions,
            index=X.index,
            name="predicted_class",
        )

    def predict_regression(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:

        logger.info(
            "Running regression inference..."
        )

        predictions = (
            self.regression_model.predict(X)
        )

        return pd.Series(
            predictions,
            index=X.index,
            name="predicted_score",
        )

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run both classification and regression.
        """

        logger.info(
            "Running full inference..."
        )

        classification = (
            self.predict_classification(X)
        )

        regression = (
            self.predict_regression(X)
        )

        results = pd.concat(
            [
                classification,
                regression,
            ],
            axis=1,
        )

        logger.info(
            "Inference completed."
        )

        return results