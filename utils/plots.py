"""Functions useful for plotting."""

import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost.sklearn import XGBRegressor


def make_donut_chart(
    df: pd.DataFrame,
    col_name: str,
    save_path=None
) -> plt.pie:
    """Create a donut chart."""
    # Create data
    size_of_groups = [x for x in df[col_name].value_counts()]
    total = np.sum(size_of_groups)

    # Get names for labels
    names = [x for x in df[col_name].value_counts().keys()]
    percentages = []
    for i in range(0, len(names)):
        percentages.append(
            f"{names[i]} : {round(size_of_groups[i]/total*100,2):,}%")

    # Create a pieplot
    plt.pie(size_of_groups, labels=percentages)
    plt.suptitle(f"Distribution in {col_name}", fontsize=16)

    # Add a circle at the center to transform it in a donut chart
    my_circle = plt.Circle((0, 0), 0.7, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)

    # display and save
    if save_path:
        plt.savefig(f"{save_path}\\donut_chart_{col_name}.jpeg")
    plt.show()
    plt.clf()


def plot_all_donuts(
    df: pd.DataFrame,
    cols_to_drop: List[str],
    save_path=None
) -> plt.pie:
    """Plots several donuts for different features."""
    # Get all colnames and filter them
    col_names = list(df.columns)
    for col in cols_to_drop:
        if col in col_names:
            col_names.remove(col)

    for col_name in col_names:
        make_donut_chart(df, col_name, save_path)


def make_correlation_matrix(
    df: pd.DataFrame,
    dataName: str,
    save_path=None
) -> sns.heatmap:
    """Build a Pearson's Correlation Matrix."""
    # get matrix
    correlation_mat = df.corr()

    # define size
    plt.figure(figsize=(18, 12))

    # draw heatmap
    ax = sns.heatmap(correlation_mat, annot=True, linewidths=.5)

    # decorate plot
    plt.title(f"Correlation Matrix {dataName}", fontsize=24)
    plt.xlabel("Image Tabular Features", fontsize=12)
    plt.ylabel("Image Tabular Features", fontsize=12)

    # display and save
    if save_path:
        plt.savefig(f"{save_path}\\corr_matrix_{dataName}.jpeg")
    plt.show()
    plt.clf()


def get_xgb_feature_importance(
    model: XGBRegressor,
    modelName: str,
    df: pd.DataFrame,
    path: str
) -> plt.figure:
    """Plot an XGBoost model's feature importances."""
    importance = pd.DataFrame({
        'features': df.columns,
        'importance': model.feature_importances_
    })

    importance.sort_values(by='importance', inplace=True)
    threshold = 0.005
    importance = importance[importance['importance'] >= threshold]

    plt.figure(figsize=(12, 16))
    plt.barh(importance['features'], importance['importance'])
    plt.title(f'{modelName} Feature Importance')
    plt.savefig(f"{path}/{modelName}_features.png", dpi=300)
    plt.show()
    plt.clf()
