"""
Classification Trainer
"""

from sklearn.pipeline import Pipeline

from .model_factory import ModelFactory


class ClassificationPipeline:

    def __init__(
        self,
        transformer,
        model_name="random_forest",
    ):

        self.pipeline = Pipeline(

            [

                (
                    "preprocessor",
                    transformer,
                ),

                (
                    "model",
                    ModelFactory.classification(
                        model_name
                    ),
                ),

            ]

        )

    def fit(self, X, y):

        self.pipeline.fit(X, y)

    def predict(self, X):

        return self.pipeline.predict(X)

    def predict_proba(self, X):

        return self.pipeline.predict_proba(X)