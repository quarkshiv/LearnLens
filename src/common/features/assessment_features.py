import numpy as np
import pandas as pd

GROUP_COLS = [
    "id_student",
    "code_module",
    "code_presentation",
]


def build_assessment_features(
    student_assessments: pd.DataFrame,
    assessments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build assessment-based features.

    Parameters
    ----------
    student_assessments : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = student_assessments.copy()

    df = df.merge(
    assessments[
        [
            "id_assessment",
            "code_module",
            "code_presentation",
        ]
    ],
    on="id_assessment",
    how="left",
)

    # --------------------------------------------------
    # Pass / Fail
    # --------------------------------------------------

    df["passed"] = (df["score"] >= 40).astype(int)

    # --------------------------------------------------
    # Statistical Features
    # --------------------------------------------------

    features = (
        df.groupby(GROUP_COLS)
        .agg(
            assessment_count=("score", "count"),
            total_score=("score", "sum"),
            average_score=("score", "mean"),
            median_score=("score", "median"),
            max_score=("score", "max"),
            min_score=("score", "min"),
            std_score=("score", "std"),
            passed_assessments=("passed", "sum"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Failed Assessments
    # --------------------------------------------------

    features["failed_assessments"] = (
        features["assessment_count"]
        - features["passed_assessments"]
    )

    # --------------------------------------------------
    # Pass Rate
    # --------------------------------------------------

    features["pass_rate"] = (
        features["passed_assessments"]
        / features["assessment_count"]
    )

    # --------------------------------------------------
    # Score Range
    # --------------------------------------------------

    features["score_range"] = (
        features["max_score"]
        - features["min_score"]
    )

    # --------------------------------------------------
    # Coefficient of Variation
    # --------------------------------------------------

    features["cv_score"] = (
        features["std_score"]
        /
        (features["average_score"] + 1e-6)
    )

    # --------------------------------------------------
    # First / Last Score
    # --------------------------------------------------

    ordered = df.sort_values("date_submitted")

    first_last = (
        ordered.groupby(GROUP_COLS)
        .agg(
            first_score=("score", "first"),
            last_score=("score", "last"),
        )
        .reset_index()
    )

    features = features.merge(
        first_last,
        on=GROUP_COLS,
        how="left",
    )

    # --------------------------------------------------
    # Improvement
    # --------------------------------------------------

    features["score_improvement"] = (
        features["last_score"]
        - features["first_score"]
    )

    # --------------------------------------------------
    # Cleaning
    # --------------------------------------------------

    features.replace(
        [np.inf, -np.inf],
        0,
        inplace=True,
    )

    features.fillna(0, inplace=True)

    return features