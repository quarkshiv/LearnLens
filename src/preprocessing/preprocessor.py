"""
Data Preprocessor
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:

    CLASSIFICATION_TARGET = "final_result"

    REGRESSION_TARGET = "final_score"

    DROP_COLUMNS = [
        "id_student",
    ]

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

    def clean(self) -> pd.DataFrame:

        logger.info("Cleaning dataset...")

        self._drop_columns()

        self._remove_duplicates()

        self._fill_missing()

        return self.df

    def classification_dataset(self):

        X = self.df.drop(
            columns=[
                self.CLASSIFICATION_TARGET,
                self.REGRESSION_TARGET,
            ],
            errors="ignore",
        )

        y = self.df[self.CLASSIFICATION_TARGET]

        return X, y

    def regression_dataset(self):

        X = self.df.drop(
            columns=[
                self.CLASSIFICATION_TARGET,
                self.REGRESSION_TARGET,
            ],
            errors="ignore",
        )

        y = self.df[self.REGRESSION_TARGET]

        return X, y

    def _drop_columns(self):

        cols = [
            c
            for c in self.DROP_COLUMNS
            if c in self.df.columns
        ]

        self.df.drop(
            columns=cols,
            inplace=True,
        )

    def _remove_duplicates(self):

        self.df.drop_duplicates(
            inplace=True,
        )

    def _fill_missing(self):

        numeric = self.df.select_dtypes(
            include="number"
        ).columns

        categorical = self.df.select_dtypes(
            exclude="number"
        ).columns

        self.df[numeric] = (
            self.df[numeric]
            .fillna(0)
        )

        self.df[categorical] = (
            self.df[categorical]
            .fillna("Unknown")
        )