"""
Encoding Pipeline
"""

from __future__ import annotations

import logging

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class EncoderBuilder:

    @staticmethod
    def build(X: pd.DataFrame):

        categorical = X.select_dtypes(
            include=[
                "object",
                "category",
            ]
        ).columns.tolist()

        numerical = X.select_dtypes(
            exclude=[
                "object",
                "category",
            ]
        ).columns.tolist()

        numeric_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numeric_pipeline,
                    numerical,
                ),
                (
                    "cat",
                    categorical_pipeline,
                    categorical,
                ),
            ]
        )

        logger.info(
            "Numerical Features : %d",
            len(numerical),
        )

        logger.info(
            "Categorical Features : %d",
            len(categorical),
        )

        return transformer