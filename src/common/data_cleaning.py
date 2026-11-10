import pandas as pd


def missing_report(df: pd.DataFrame) -> pd.DataFrame:

    report = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing %": (df.isnull().mean() * 100).round(2).values
    })

    report = report.sort_values("Missing %", ascending=False)

    return report