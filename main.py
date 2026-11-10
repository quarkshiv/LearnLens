from __future__ import annotations

import logging
import time

from src.pipelines.training_pipeline import TrainingPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


logger = logging.getLogger(__name__)


def main():

    logger.info("=" * 80)
    logger.info("Student Performance Prediction")
    logger.info("=" * 80)

    start = time.perf_counter()

    pipeline = TrainingPipeline()

    results = pipeline.run()

    elapsed = time.perf_counter() - start

    logger.info("=" * 80)
    logger.info("Training Finished Successfully")
    logger.info(
        "Elapsed Time: %.2f seconds",
        elapsed,
    )
    logger.info("=" * 80)

    return results


if __name__ == "__main__":

    main()