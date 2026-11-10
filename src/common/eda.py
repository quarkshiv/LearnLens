import pandas as pd


def dataset_summary(df: pd.DataFrame, name: str) -> pd.DataFrame:

    summary = {
        "Dataset": name,
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Duplicates": df.duplicated().sum(),
        "Missing Cells": df.isnull().sum().sum(),
        "Memory (MB)": round(df.memory_usage(deep=True).sum()/1024**2,2)
    }

    return pd.DataFrame([summary])

def column_summary(df):

    summary = pd.DataFrame({

        "Column":df.columns,

        "Data Type":df.dtypes.astype(str),

        "Missing":df.isnull().sum().values,

        "Unique":df.nunique().values

    })

    return summary