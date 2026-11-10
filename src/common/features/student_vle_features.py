"""
Student VLE Feature Engineering

This module builds all engagement-related features from the
student_vle and vle datasets.

"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from .feature_config import (
    ACTIVITY_WEIGHTS,
    EARLY_PERIOD,
    MID_PERIOD,
    GROUP_COLS,
    EPSILON,
)

logger = logging.getLogger(__name__)


def build_student_vle_features(
    student_vle: pd.DataFrame,
    vle: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build engagement, temporal, activity,
    and weighted interaction features.

    Parameters
    ----------
    student_vle : pd.DataFrame
    vle : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    logger.info("Building Student VLE Features...")

    # -------------------------------------------------------
    # Merge Activity Types
    # -------------------------------------------------------

    df = student_vle.merge(
        vle[
            [
                "id_site",
                "code_module",
                "code_presentation",
                "activity_type",
            ]
        ],
        on=[
            "id_site",
            "code_module",
            "code_presentation",
        ],
        how="left",
    )

    # -------------------------------------------------------
    # Missing Activity Types
    # -------------------------------------------------------

    df["activity_type"] = (
        df["activity_type"]
        .fillna("unknown")
        .str.lower()
    )

    # -------------------------------------------------------
    # Activity Weights
    # -------------------------------------------------------

    df["activity_weight"] = (
        df["activity_type"]
        .map(ACTIVITY_WEIGHTS)
        .fillna(1.0)
    )

    df["weighted_clicks"] = (
        df["sum_click"]
        * df["activity_weight"]
    )

    # -------------------------------------------------------
    # Daily Aggregation
    # -------------------------------------------------------

    daily = (
        df.groupby(
            GROUP_COLS + ["date"],
            as_index=False,
        )
        .agg(
            daily_clicks=("sum_click", "sum"),
            daily_weighted_clicks=(
                "weighted_clicks",
                "sum",
            ),
            daily_resources=(
                "id_site",
                "nunique",
            ),
            daily_interactions=(
                "id_site",
                "count",
            ),
        )
    )

    logger.info(
        "Daily aggregation completed."
    )

    # -------------------------------------------------------
    # General Engagement
    # -------------------------------------------------------

    features = (
        daily.groupby(
            GROUP_COLS,
            as_index=False,
        )
        .agg(
            total_clicks=(
                "daily_clicks",
                "sum",
            ),
            weighted_clicks=(
                "daily_weighted_clicks",
                "sum",
            ),
            study_days=(
                "date",
                "nunique",
            ),
            total_interactions=(
                "daily_interactions",
                "sum",
            ),
            active_weeks=(
                "date",
                lambda x: (
                    (x // 7)
                    .nunique()
                ),
            ),
            avg_daily_clicks=(
                "daily_clicks",
                "mean",
            ),
            median_daily_clicks=(
                "daily_clicks",
                "median",
            ),
            max_daily_clicks=(
                "daily_clicks",
                "max",
            ),
            std_daily_clicks=(
                "daily_clicks",
                "std",
            ),
        )
    )

    logger.info(
        "General engagement features completed."
    )
    
    # -------------------------------------------------------
    # Resource Features
    # -------------------------------------------------------

    resource_features = (
        df.groupby(
            GROUP_COLS,
            as_index=False,
        )
        .agg(
            unique_resources=(
                "id_site",
                "nunique",
            ),
            unique_activity_types=(
                "activity_type",
                "nunique",
            ),
        )
    )

    features = features.merge(
        resource_features,
        on=GROUP_COLS,
        how="left",
    )

    features["clicks_per_resource"] = (
        features["total_clicks"]
        /
        (
            features["unique_resources"]
            + EPSILON
        )
    )

    features["resource_diversity"] = (
        features["unique_activity_types"]
        /
        (
            features["unique_resources"]
            + EPSILON
        )
    )

    # -------------------------------------------------------
    # Activity Distribution
    # -------------------------------------------------------

    activity = (
        df.groupby(
            GROUP_COLS + ["activity_type"],
            as_index=False,
        )["sum_click"]
        .sum()
    )

    pivot = (
        activity.pivot_table(
            index=GROUP_COLS,
            columns="activity_type",
            values="sum_click",
            fill_value=0,
        )
        .reset_index()
    )

    # -------------------------------------------------------
    # Important Activity Ratios
    # -------------------------------------------------------

    IMPORTANT_ACTIVITIES = {
        "homepage": "homepage_ratio",
        "forumng": "forum_ratio",
        "quiz": "quiz_ratio",
        "resource": "resource_ratio",
        "oucontent": "content_ratio",
        "subpage": "subpage_ratio",
        "page": "page_ratio",
        "url": "url_ratio",
        "ouwiki": "wiki_ratio",
        "oucollaborate": "collaboration_ratio",
    }

    for activity_name, feature_name in IMPORTANT_ACTIVITIES.items():

        if activity_name not in pivot.columns:
            pivot[activity_name] = 0

        pivot[feature_name] = (
            pivot[activity_name]
            /
            (
                pivot.drop(
                    columns=GROUP_COLS,
                    errors="ignore",
                )
                .sum(axis=1)
                + EPSILON
            )
        )

    ratio_columns = (
        GROUP_COLS
        +
        list(IMPORTANT_ACTIVITIES.values())
    )

    features = features.merge(
        pivot[ratio_columns],
        on=GROUP_COLS,
        how="left",
    )

    logger.info(
        "Activity distribution completed."
    )

    # -------------------------------------------------------
    # Weighted Features
    # -------------------------------------------------------

    features["weighted_daily_clicks"] = (
        features["weighted_clicks"]
        /
        (
            features["study_days"]
            + EPSILON
        )
    )

    features["activity_density"] = (
        features["total_interactions"]
        /
        (
            features["study_days"]
            + EPSILON
        )
    )

    features["clicks_per_day"] = (
        features["total_clicks"]
        /
        (
            features["study_days"]
            + EPSILON
        )
    )

    # -------------------------------------------------------
    # Temporal Features
    # -------------------------------------------------------

    temporal = (
        daily.groupby(
            GROUP_COLS,
            as_index=False,
        )
        .agg(
            first_activity=(
                "date",
                "min",
            ),
            last_activity=(
                "date",
                "max",
            ),
        )
    )

    temporal["engagement_duration"] = (
        temporal["last_activity"]
        -
        temporal["first_activity"]
    )

    features = features.merge(
        temporal,
        on=GROUP_COLS,
        how="left",
    )

    # -------------------------------------------------------
    # Early / Mid / Late Engagement
    # -------------------------------------------------------

    early = (
        df.loc[
            df["date"] <= EARLY_PERIOD
        ]
        .groupby(
            GROUP_COLS,
            as_index=False,
        )["sum_click"]
        .sum()
        .rename(
            columns={
                "sum_click": "early_clicks"
            }
        )
    )

    mid = (
        df.loc[
            (df["date"] > EARLY_PERIOD)
            &
            (df["date"] <= MID_PERIOD)
        ]
        .groupby(
            GROUP_COLS,
            as_index=False,
        )["sum_click"]
        .sum()
        .rename(
            columns={
                "sum_click": "mid_clicks"
            }
        )
    )

    late = (
        df.loc[
            df["date"] > MID_PERIOD
        ]
        .groupby(
            GROUP_COLS,
            as_index=False,
        )["sum_click"]
        .sum()
        .rename(
            columns={
                "sum_click": "late_clicks"
            }
        )
    )

        # -------------------------------------------------------
    # Merge Early / Mid / Late Features
    # -------------------------------------------------------

    for part in (early, mid, late):
        features = features.merge(
            part,
            on=GROUP_COLS,
            how="left",
        )

    features[
        [
            "early_clicks",
            "mid_clicks",
            "late_clicks",
        ]
    ] = features[
        [
            "early_clicks",
            "mid_clicks",
            "late_clicks",
        ]
    ].fillna(0)

    features["early_late_ratio"] = (
        features["early_clicks"]
        /
        (
            features["late_clicks"]
            + EPSILON
        )
    )

    # -------------------------------------------------------
    # Session Statistics
    # -------------------------------------------------------

    logger.info(
        "Computing session statistics..."
    )

    session_df = (
        daily[
            GROUP_COLS + ["date"]
        ]
        .sort_values(
            GROUP_COLS + ["date"]
        )
        .copy()
    )

    session_df["gap"] = (
        session_df
        .groupby(GROUP_COLS)["date"]
        .diff()
    )

    gap_features = (
        session_df.groupby(
            GROUP_COLS,
            as_index=False,
        )
        .agg(
            avg_gap_between_sessions=(
                "gap",
                "mean",
            ),
            longest_gap=(
                "gap",
                "max",
            ),
        )
    )

    features = features.merge(
        gap_features,
        on=GROUP_COLS,
        how="left",
    )

    # -------------------------------------------------------
    # Additional Behaviour Indicators
    # -------------------------------------------------------

    features["interaction_per_resource"] = (
        features["total_interactions"]
        /
        (
            features["unique_resources"]
            + EPSILON
        )
    )

    features["weighted_resource_usage"] = (
        features["weighted_clicks"]
        /
        (
            features["unique_resources"]
            + EPSILON
        )
    )

    features["engagement_intensity"] = (
        features["weighted_clicks"]
        /
        (
            features["engagement_duration"]
            + 1
        )
    )

    features["daily_consistency"] = (
        features["study_days"]
        /
        (
            features["engagement_duration"]
            + 1
        )
    )

    # -------------------------------------------------------
    # Replace Invalid Values
    # -------------------------------------------------------

    features.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    # -------------------------------------------------------
    # Fill Missing Values
    # -------------------------------------------------------

    numeric_cols = features.select_dtypes(
        include=[
            np.number,
        ]
    ).columns

    features[numeric_cols] = (
        features[numeric_cols]
        .fillna(0)
    )

    # -------------------------------------------------------
    # Sort Columns
    # -------------------------------------------------------

    features = features.sort_values(
        GROUP_COLS
    ).reset_index(
        drop=True
    )

    logger.info(
        "Student VLE Feature Engineering Completed."
    )

    logger.info(
        "Generated %d features.",
        features.shape[1] - len(GROUP_COLS),
    )

    return features
