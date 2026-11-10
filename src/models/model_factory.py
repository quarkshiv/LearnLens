"""
Model Factory

Create machine learning models from a single interface.
"""

from __future__ import annotations

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


class ModelFactory:

    @staticmethod
    def classification(name: str):

        models = {

            "logistic": LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),

            "decision_tree": DecisionTreeClassifier(
                random_state=42,
            ),

            "random_forest": RandomForestClassifier(
              n_estimators=300,
              max_depth=20,
              min_samples_split=5,
              min_samples_leaf=2,
              random_state=42,
              n_jobs=-1,
            ),

        }

        return models[name]

    @staticmethod
    def regression(name: str):

        models = {

            "linear": LinearRegression(),

            "decision_tree": DecisionTreeRegressor(
                random_state=42,
            ),

            "random_forest": RandomForestRegressor(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
            ),

        }

        return models[name]