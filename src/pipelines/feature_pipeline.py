from pathlib import Path
import logging

from .data_pipeline import DataPipeline
from src.common.features.master_builder import MasterTableBuilder


logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    End-to-end Feature Engineering Pipeline.

    Responsibilities:
        - Load raw datasets
        - Merge datasets
        - Build feature tables
        - Validate features
        - Save master dataset
        - Print summary
    """

    def __init__(self):
        self.datasets = None
        self.master = None

    def run(self):

        logger.info("=" * 70)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 70)

        # --------------------------------------------------
        # Load Data
        # --------------------------------------------------

        logger.info("Loading datasets...")

        data_pipeline = DataPipeline()

        self.datasets, _ = data_pipeline.run()

        logger.info("Datasets loaded successfully.")

        # --------------------------------------------------
        # Build Master Feature Table
        # --------------------------------------------------

        logger.info("Building master feature table...")

        builder = MasterTableBuilder(
            datasets=self.datasets
        )

        self.master = builder.build()

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        builder.summary()

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        builder.save()

        logger.info("Master feature table saved successfully.")

        logger.info("=" * 70)
        logger.info("Feature Engineering Pipeline Finished")
        logger.info("=" * 70)

        return self.master