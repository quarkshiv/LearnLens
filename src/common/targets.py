"""
Target Builders

Build classification and regression targets.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

GROUP_COLS = [
    "id_student",
    "code_module",
    "code_presentation",
]


def build_regression_target(
    student_assessments: pd.DataFrame,
    assessments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a weighted final score for each student.

    Parameters
    ----------
    student_assessments : DataFrame
    assessments : DataFrame

    Returns
    -------
    DataFrame
        id_student
        code_module
        code_presentation
        final_score
    """

    logger.info("Building regression target...")

    df = student_assessments.merge(
    assessments[
        [
            "id_assessment",
            "code_module",
            "code_presentation",
            "weight",
        ]
    ],
    on="id_assessment",
    how="left",
)

    df["weight"] = df["weight"].fillna(0)

    df["weighted_score"] = (
        df["score"]
        * df["weight"]
    )

    result = (
        df.groupby(
            GROUP_COLS,
            as_index=False,
        )
        .agg(
            total_weight=("weight", "sum"),
            total_score=("weighted_score", "sum"),
        )
    )

    result["final_score"] = (
        result["total_score"]
        /
        result["total_weight"]
    )

    result["final_score"] = (
        result["final_score"]
        .fillna(0)
    )

    result = result[
        GROUP_COLS
        + [
            "final_score",
        ]
    ]

    logger.info(
        "Regression target built successfully."
    )

    return result

def build_classification_target(
    student_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the classification target.

    Parameters
    ----------
    student_info : DataFrame

    Returns
    -------
    DataFrame
        id_student
        code_module
        code_presentation
        final_result
    """

    logger.info("Building classification target...")

    target = student_info[
        [
            "id_student",
            "code_module",
            "code_presentation",
            "final_result",
        ]
    ].copy()

    logger.info(
        "Classification target built successfully."
    )

    return target