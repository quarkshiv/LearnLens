"""
Model Persistence Utilities

Save and load trained models and preprocessing objects.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib


logger = logging.getLogger(__name__)


MODEL_DIR = Path("artifacts")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


class ModelIO:
    """
    Save and load models using Joblib.
    """

    @staticmethod
    def save(
        obj,
        filename: str,
    ) -> Path:

        path = MODEL_DIR / filename

        joblib.dump(
            obj,
            path,
        )

        logger.info(
            "Saved: %s",
            path,
        )

        return path

    @staticmethod
    def load(
        filename: str,
    ):

        path = MODEL_DIR / filename

        if not path.exists():

            raise FileNotFoundError(
                f"Model not found: {path}"
            )

        logger.info(
            "Loaded: %s",
            path,
        )

        return joblib.load(path)

    @staticmethod
    def save_classification_model(
        model,
    ):

        return ModelIO.save(
            model,
            "classification_model.pkl",
        )

    @staticmethod
    def save_regression_model(
        model,
    ):

        return ModelIO.save(
            model,
            "regression_model.pkl",
        )

    @staticmethod
    def save_transformer(
        transformer,
    ):

        return ModelIO.save(
            transformer,
            "transformer.pkl",
        )

    @staticmethod
    def load_classification_model():

        return ModelIO.load(
            "classification_model.pkl",
        )

    @staticmethod
    def load_regression_model():

        return ModelIO.load(
            "regression_model.pkl",
        )

    @staticmethod
    def load_transformer():

        return ModelIO.load(
            "transformer.pkl",
        )