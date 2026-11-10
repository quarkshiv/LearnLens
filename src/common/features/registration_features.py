import numpy as np
import pandas as pd


GROUP_COLS = [
    "id_student",
    "code_module",
    "code_presentation",
]


def build_registration_features(
    student_registration: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build registration-related features.

    Parameters
    ----------
    student_registration : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = student_registration.copy()

    # ----------------------------------------------
    # Withdrawal
    # ----------------------------------------------

    df["is_withdrawn"] = (
        df["date_unregistration"]
        .notna()
        .astype(int)
    )

    # ----------------------------------------------
    # Registration Timing
    # ----------------------------------------------

    df["registered_before_start"] = (
        df["date_registration"] <= 0
    ).astype(int)

    df["registration_day"] = df["date_registration"]

    df["registration_delay"] = (
        df["date_registration"]
        .clip(lower=0)
    )

    # ----------------------------------------------
    # Registration Categories
    # ----------------------------------------------

    df["early_registration"] = (
        df["date_registration"] <= -30
    ).astype(int)

    df["late_registration"] = (
        df["date_registration"] > 0
    ).astype(int)

    # ----------------------------------------------
    # Enrollment Lead Time
    # ----------------------------------------------

    df["days_before_start"] = (
        df["date_registration"]
        .clip(upper=0)
        .abs()
    )

    # ----------------------------------------------
    # Study Duration
    # ----------------------------------------------

    df["study_duration"] = (
        df["date_unregistration"]
        - df["date_registration"]
    )

    df["study_duration"] = (
        df["study_duration"]
        .fillna(0)
    )

    # ----------------------------------------------
    # Clean
    # ----------------------------------------------

    df.replace(
        [np.inf, -np.inf],
        0,
        inplace=True,
    )

    df.fillna(0, inplace=True)

    return df