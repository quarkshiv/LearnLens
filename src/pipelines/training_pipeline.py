"""
Training Pipeline
"""

from __future__ import annotations

import logging

from src.common.data_loader import load_all_data
from src.common.features.master_builder import MasterTableBuilder

from src.preprocessing.preprocessor import DataPreprocessor
from src.preprocessing.encoder import EncoderBuilder
from src.preprocessing.splitter import DataSplitter

from src.models.benchmark import Benchmark

from src.utils.model_io import ModelIO


logger = logging.getLogger(__name__)


class TrainingPipeline:

    def run(self):

        logger.info("=" * 70)
        logger.info("START TRAINING PIPELINE")
        logger.info("=" * 70)

        # --------------------------------------------------
        # Load datasets
        # --------------------------------------------------

        datasets = load_all_data()

        # --------------------------------------------------
        # Build master dataset
        # --------------------------------------------------

        builder = MasterTableBuilder(datasets)

        master = builder.build()

        builder.summary()

        builder.save()

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------

        preprocessor = DataPreprocessor(master)

        master = preprocessor.clean()

        # --------------------------------------------------
        # Classification Dataset
        # --------------------------------------------------

        X_cls, y_cls = preprocessor.classification_dataset()

        # --------------------------------------------------
        # Regression Dataset
        # --------------------------------------------------

        X_reg, y_reg = preprocessor.regression_dataset()

        # --------------------------------------------------
        # Encoder
        # --------------------------------------------------

        transformer = EncoderBuilder.build(X_cls)

        # --------------------------------------------------
        # Split
        # --------------------------------------------------

        (
            X_train_cls,
            X_test_cls,
            y_train_cls,
            y_test_cls,
        ) = DataSplitter.classification_split(
            X_cls,
            y_cls,
        )

        (
            X_train_reg,
            X_test_reg,
            y_train_reg,
            y_test_reg,
        ) = DataSplitter.regression_split(
            X_reg,
            y_reg,
        )

        # --------------------------------------------------
        # Benchmark
        # --------------------------------------------------

        results = Benchmark.run_all(

            transformer,

            X_train_cls,
            X_test_cls,
            y_train_cls,
            y_test_cls,

            X_train_reg,
            X_test_reg,
            y_train_reg,
            y_test_reg,

        )

        # --------------------------------------------------
        # Save Models
        # --------------------------------------------------

        ModelIO.save_classification_model(
            results["classification_model"]
        )

        ModelIO.save_regression_model(
            results["regression_model"]
        )
        """
        ModelIO.save_transformer(
            transformer
        )
        """
        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 70)

        return results