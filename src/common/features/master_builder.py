"""
Master Feature Table Builder.

This module combines all engineered features into a single
machine learning table.
"""

from __future__ import annotations

import logging

import pandas as pd
from pathlib import Path
from .assessment_features import build_assessment_features
from .registration_features import build_registration_features
from .student_vle_features import build_student_vle_features
from src.common.targets import (
    build_regression_target,
    build_classification_target,
)
logger = logging.getLogger(__name__)


class MasterTableBuilder:
    """
    Build the final master feature table.
    """

    def __init__(self, datasets: dict[str, pd.DataFrame]) -> None:
        self.datasets = datasets

    def build(self) -> pd.DataFrame:

        logger.info("=" * 60)
        logger.info("Building Master Feature Table")
        logger.info("=" * 60)

        # --------------------------------------------------
        # Base Table
        # --------------------------------------------------

        master = self.datasets["student_info"].copy()

        logger.info(
            "Base table shape: %s",
            master.shape,
        )

        # --------------------------------------------------
        # Registration Features
        # --------------------------------------------------

        registration = build_registration_features(
            self.datasets["student_registration"]
        )

        logger.info(
            "Registration features: %s",
            registration.shape,
        )

        master = master.merge(
            registration,
            on=[
                "id_student",
                "code_module",
                "code_presentation",
            ],
            how="left",
        )

        # --------------------------------------------------
        # Assessment Features
        # --------------------------------------------------

        assessment = build_assessment_features(
            self.datasets["student_assessments"],
            self.datasets["assessments"],
        )

        logger.info(
            "Assessment features: %s",
            assessment.shape,
        )

        master = master.merge(
            assessment,
            on=[
                "id_student",
                "code_module",
                "code_presentation",
            ],
            how="left",
        )


        # --------------------------------------------------
        # Student VLE Features
        # --------------------------------------------------

        vle_features = build_student_vle_features(
            self.datasets["student_vle"],
            self.datasets["vle"],
        )

        logger.info(
            "Student VLE features: %s",
            vle_features.shape,
        )

        master = master.merge(
            vle_features,
            on=[
                "id_student",
                "code_module",
                "code_presentation",
            ],
            how="left",
        )

        # --------------------------------------------------
        # Regression Target
        # --------------------------------------------------

        regression_target = build_regression_target(
            self.datasets["student_assessments"],
            self.datasets["assessments"],
        )

        master = master.merge(
            regression_target,
            on=[
                "id_student",
                "code_module",
                "code_presentation",
            ],
            how="left",
        )

        # --------------------------------------------------
        # Classification Target
        # --------------------------------------------------
        """
        classification_target = build_classification_target(
            self.datasets["student_info"],
        )

        master = master.merge(
            classification_target,
            on=[
                "id_student",
                "code_module",
                "code_presentation",
            ],
            how="left",
        )
        """
        # --------------------------------------------------
        # Fill Missing Values
        # --------------------------------------------------
        # Remove samples without targets

        master = master.dropna(
        subset=[
        "final_result",
        "final_score",
        ]
      )
        numeric_columns = master.select_dtypes(
            include="number"
        ).columns

        master[numeric_columns] = (
            master[numeric_columns]
            .fillna(0)
        )

        categorical_columns = master.select_dtypes(
            exclude="number"
        ).columns

        master[categorical_columns] = (
            master[categorical_columns]
            .fillna("Unknown")
        )

        # --------------------------------------------------
        # Remove Duplicate Rows
        # --------------------------------------------------

        master = (
            master
            .drop_duplicates()
            .reset_index(drop=True)
        )

        logger.info("=" * 60)
        logger.info(
            "Master table shape: %s",
            master.shape,
        )
        logger.info("=" * 60)
        self.master = master

        return self.master




    def summary(self) -> None:
        """
        Print a concise summary of the master dataset.
        """

        if not hasattr(self, "master"):
            raise ValueError(
                "Build the master table before calling summary()."
            )

        logger.info("=" * 60)
        logger.info("MASTER DATASET SUMMARY")
        logger.info("=" * 60)

        logger.info(
            "Shape: %s",
            self.master.shape,
        )

        logger.info(
            "Columns: %d",
            len(self.master.columns),
        )

        logger.info(
            "Missing Values: %d",
            self.master.isna().sum().sum(),
        )

        logger.info("=" * 60)


    def save(self, path: str = "outputs/master_dataset.csv") -> None:
        """Save the master dataset to a CSV file.

        Parameters
        ----------
        path : str, optional
            Output file path.
        """
        if not hasattr(self, "master"):
            raise ValueError(
                "Build the master dataset before calling save()."
            )

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.master.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Master dataset saved successfully."
        )

        logger.info(
            "Location: %s",
            output_path.resolve(),
        )

        logger.info(
            "Rows: %d | Columns: %d",
            self.master.shape[0],
            self.master.shape[1],
        )