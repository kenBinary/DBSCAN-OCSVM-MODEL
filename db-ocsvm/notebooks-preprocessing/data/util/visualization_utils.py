import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


def plot_feature_outliers(df):
    """
    Create boxplots for each numerical feature to visualize outliers
    Args:
        df (pd.DataFrame): DataFrame containing numerical features
    """
    sns.set_style("whitegrid")

    # this calculates how many rows in the subplot
    n_cols = 4
    n_rows = math.ceil(len(df.columns) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.ravel()

    # Create boxplot for each feature
    for idx, column in enumerate(df.columns):
        if idx < len(df.columns):
            sns.boxplot(
                y=df[column],
                ax=axes[idx],
                width=0.5,
                color="lightblue",
                flierprops={
                    "marker": "o",
                    "markerfacecolor": "red",
                    "markeredgecolor": "darkred",
                    "markersize": 8,
                },
            )

            # axes[idx].set_title(f"Distribution of {column}", pad=10)
            axes[idx].set_title(column, pad=10)
            axes[idx].set_ylabel("Value")

    # Removes empty subplots
    if len(df.columns) < 6:
        fig.delaxes(axes[5])

    plt.tight_layout()
    plt.show()
