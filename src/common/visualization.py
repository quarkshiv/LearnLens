import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path


def plot_missing(df,name):

    missing=df.isnull().sum()

    missing=missing[missing>0]

    if missing.empty:

        return

    plt.figure(figsize=(10,5))

    sns.barplot(x=missing.index,y=missing.values)

    plt.xticks(rotation=45)

    plt.tight_layout()

    Path("figures").mkdir(parents=True,exist_ok=True)

    plt.savefig(f"figures/{name}_missing.png")

    plt.close()