"""
Training Manager
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Trainer:

    @staticmethod
    def train(model, X_train, y_train):

        logger.info("Training started...")

        model.fit(
            X_train,
            y_train,
        )

        logger.info("Training finished.")

        return model