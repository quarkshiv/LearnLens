"""
Evaluation Report
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT = Path("outputs")

OUTPUT.mkdir(
    exist_ok=True,
)


class ReportGenerator:

    @staticmethod
    def save_metrics(
        metrics: dict,
        filename: str,
    ):

        pd.DataFrame(
            [metrics]
        ).to_csv(

            OUTPUT / filename,

            index=False,

        )

    @staticmethod
    def save_classification_report(
        report,
        filename="classification_report.csv",
    ):

        report.to_csv(

            OUTPUT / filename,

            index=True,

        )