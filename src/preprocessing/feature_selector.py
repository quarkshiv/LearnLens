"""
Feature Selection Utilities
"""

from __future__ import annotations

import logging

import pandas as pd

from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)


class FeatureSelector:

    def __init__(self):

        self.selector = VarianceThreshold(
            threshold=0.0
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
    ):

        logger.info(
            "Running Variance Threshold..."
        )

        transformed = self.selector.fit_transform(X)

        selected_columns = (
            X.columns[
                self.selector.get_support()
            ]
        )

        return pd.DataFrame(
            transformed,
            columns=selected_columns,
            index=X.index,
        )

    def transform(
        self,
        X: pd.DataFrame,
    ):

        transformed = self.selector.transform(X)

        selected_columns = (
            X.columns[
                self.selector.get_support()
            ]
        )

        return pd.DataFrame(
            transformed,
            columns=selected_columns,
            index=X.index,
        )