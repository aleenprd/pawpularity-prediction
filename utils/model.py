"""Useful functions for building/evaluating models."""


import pandas as pd
import sys
import numpy as np
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def define_global_variables() -> List:
    """Automatically define some important paths."""
    data_path = "data/"
    model_path = "model/"

    train_path = f"{data_path}train/"
    test_path = f"{data_path}test/"
    sample_path = f"{data_path}test/"
    train_file_name = "train.csv"
    test_file_name = "test.csv"

    plots_path = "plots/"
    edaplots_path = f"{plots_path}eda/"

    return [data_path, model_path, train_path, test_path,
            sample_path, train_file_name, test_file_name,
            plots_path, edaplots_path]


def regression_train_preprocess(df: pd.DataFrame) -> List:
    """Perform pre-processing for regression task."""
    df = df.set_index("Id")
    X = df.drop('Pawpularity', axis=1)
    y = df['Pawpularity']

    return [X, y]


def build_rfr_pipeline() -> Pipeline:
    """Build Random Forest Regression pipeline."""
    random_grid_random_forests = {
        # Number of trees in random fores
        'n_estimators': [int(x) for x in np.linspace(
            start=200, stop=2000, num=10)],
        # Number of features to consider at every split
        'max_features': ['auto', 'sqrt'],
        # Maximum number of levels in tree
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
        # Minimum number of samples required to split a node
        'min_samples_split': [2, 5, 10],
        # Minimum number of samples required at each leaf node
        'min_samples_leaf': [1, 2, 3, 4],
        # Method of selecting samples for training each tree
        'bootstrap': [True, False]
                                }
    print(random_grid_random_forests)
    print()

    random_forest_regression = Pipeline(
        [
            ('model', RandomizedSearchCV(
                estimator=RandomForestRegressor(),
                param_distributions=random_grid_random_forests,
                n_iter=100,
                cv=3,
                verbose=2,
                random_state=42,
                n_jobs=-1))
        ]
    )

    return random_forest_regression


def build_xgbr_pipeline() -> Pipeline:
    """Build XGBoost Regression pipeline."""
    random_grid_xgboost = {
        'n_estimators': [int(x) for x in np.linspace(
            start=200, stop=2000, num=10)],
        'learning_rate': [round(float(x), 2) for x in np.linspace(
            start=0.01, stop=0.51, num=10)],
        'subsample': [round(float(x), 2) for x in np.linspace(
            start=0.3, stop=0.6, num=4)],
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'colsample_bytree': [round(float(x), 2) for x in np.linspace(
            start=0.2, stop=0.5, num=4)],
        'min_child_weight': [1, 2, 3, 4]
                        }
    print(random_grid_xgboost)
    print()

    xgb_regression = Pipeline(
        [
            ('model', RandomizedSearchCV(
                estimator=XGBRegressor(),
                param_distributions=random_grid_xgboost,
                cv=5,
                n_iter=100,
                scoring='neg_root_mean_squared_error',
                error_score=0,
                verbose=3,
                n_jobs=-1))
        ]
    )

    return xgb_regression


def print_score_statistics(df: pd.DataFrame, df_name: str) -> None:
    """Print statistics on the pwpularity score."""
    if isinstance(df, pd.DataFrame):
        print(f"AVG {df_name} Set Score : {round(np.mean(df.Pawpularity),3)}")
        # AVG : 20.59
        print(
            f"STDEV. {df_name} Set Score : {round(np.std(df.Pawpularity),3)}")
        # STDEV : 20.59
    else:
        print(f"AVG {df_name} Set Score : {round(np.mean(df),3)}")
        # AVG : 20.59
        print(f"STDEV. {df_name} Set Score : {round(np.std(df),3)}")
        # STDEV : 20.59
    print()
