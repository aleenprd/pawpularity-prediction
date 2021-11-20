"""Pipeline for regressing tabular data."""


import pickle
import pandas as pd
import numpy as np
import math
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from utils import misc
from utils import plots
from utils import model


warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # DEFINE GLOBAL VARIABLES
    data_path, model_path, train_path, test_path, sample_path, \
        train_file_name, test_file_name, plotsPath, eda_plots_path = \
        model.define_global_variables()

    # LOAD TABULAR DATA
    data_type = {
        'Id': 'string',
        'Subject Focus': np.uint8, 'Eyes': np.uint8, 'Face': np.uint8,
        'Near': np.uint8, 'Action': np.uint8, 'Accessory': np.uint8,
        'Group': np.uint8, 'Collage': np.uint8, 'Human': np.uint8,
        'Occlusion': np.uint8, 'Info': np.uint8, 'Blur': np.uint8,
        'Pawpularity': np.uint8,
    }

    train = pd.read_csv(f"{data_path}/{train_file_name}", dtype=data_type)
    test = pd.read_csv(f"{data_path}/{test_file_name}", dtype=data_type)

    # EXPLORATORY DATA ANALYSIS
    for df in [train, test]:
        misc.explore_data(df)
        plots.plot_all_donuts(
            df, cols_to_drop=["Id", "Pawpularity"], save_path=eda_plots_path)

    plots.make_correlation_matrix(
        train, dataName="Train Set", save_path=eda_plots_path)

    # REGRESSION PREDICTIVE ANALYSIS

    # PREPROCESS DATA FOR REGRESSOR MODELS
    X, y = model.regression_train_preprocess(train)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"Train Set Shape : {X_train.shape}")
    print(f"Validation Set Shape : {X_val.shape}")

    # LOAD MODELS (OPTIONAL : TUNE, TRAIN, SAVE)
    if Path(f"{model_path}/random_forest_regression.pkl").is_file():
        random_forest_regression = pickle.load(open(
            f"{model_path}/random_forest_regression.pkl", 'rb'))
    else:
        random_forest_regression = model.build_rfr_pipeline().fit(X_train, y_train)
        pickle.dump(random_forest_regression, open(
            f"{model_path}/random_forest_regression.pkl", 'wb'))

    if Path(f"{model_path}/xgb_regression.pkl").is_file():
        xgb_regression = pickle.load(open(
            f"{model_path}/xgb_regression.pkl", 'rb'))
    else:
        xgb_regression = model.build_xgbr_pipeline().fit(X_train, y_train)
        pickle.dump(xgb_regression, open(
            f"{model_path}/xgb_regression.pkl", 'wb'))

    # EVALUATE MODELS' PERFORMANCE ON VALIDATION SET

    # These are the most important features the models consider for
    # splitting the data and building leaves and branches
    xgb_model_rand_cv = xgb_regression['model']
    xgb_model = xgb_model_rand_cv.best_estimator_
    plots.get_xgb_feature_importance(
        xgb_model, "XGBoost", X_train, eda_plots_path)

    rf_model_rand_cv = random_forest_regression['model']
    rf_model = rf_model_rand_cv.best_estimator_
    plots.get_xgb_feature_importance(
        rf_model, "RandomForests", X_train, eda_plots_path)

    # Our models are stupid and are basically almost as if
    # predicting the mean all the time
    # We really need to include data on the images as engineered features
    # A Neural Network could probably learn better than these methods,
    # but still...

    y_naive_scores = [np.mean(y_val)] * len(y_val + 1)
    rf_regression_scores = random_forest_regression.predict(X_val)
    xgb_regressionScores = xgb_regression.predict(X_val)

    model.print_score_statistics(y, "Full Dataset")
    model.print_score_statistics(y_train, "Train")
    model.print_score_statistics(y_val, "Validation")

    model.print_score_statistics(y_naive_scores, "Naive Method")
    model.print_score_statistics(pd.Series(rf_regression_scores), "RFR Scores")
    model.print_score_statistics(xgb_regressionScores, "XGB Scores")

    # RMSE is the King metric in this contest and here's how our models compare
    # to the 'naive' baseline
    print(
        f"RMSE Naive Method : {round(math.sqrt(mean_squared_error(y_val, y_naive_scores)),3)}"
        )  # 21.025
    print(
        f"RMSE Random Forests : {round(math.sqrt(mean_squared_error(y_val, rf_regression_scores)),3)}"
        )  # 21.03
    print(
        f"RMSE XGBoostt : {round(math.sqrt(mean_squared_error(y_val, xgb_regressionScores)),3)}"
        )  # 21.00
    print()

    # Best possible score is 1.0 and it can be negative
    # (because the model can be arbitrarily worse).
    # A constant model that always predicts the expected value of y,
    # disregarding the input features,
    # would get a  score of 0.0.

    print(
        f"R2 Naive Method : {round(r2_score(y_val, y_naive_scores),3)}")  # 0.0
    print(
        f"R2 Random Forests : {round(r2_score(y_val, rf_regression_scores),3)}")  # -0.001
    print(
        f"R2 XGBoost : {round(r2_score(y_val, xgb_regressionScores),3)}")  # 0.002

    # The distributions in the predicted
    # scores are mostly around the mean

    plt.figure(figsize=(16, 12))
    plt.title('Red: RFR Scores vs Green: XGB Scores')
    sns.histplot(rf_regression_scores, color="red", alpha=0.25)
    sns.histplot(xgb_regressionScores, color="green", alpha=0.5)
    plt.xlim(30, 45.0)
    plt.legend(prop={'size': 16})
    plt.show()
    plt.clf()

    # The distributions in the train
    # and val are normal
    plt.figure(figsize=(16, 12))
    plt.title('Red: Validation Scores vs Yellow: Train Scores')
    sns.histplot(y_val, color="red", alpha=0.7)
    sns.histplot(y_train, color="yellow", alpha=0.25)
    plt.show()
    plt.clf()

    plt.figure(figsize=(16, 12))
    plt.title('Overlay of All Scores')
    sns.histplot(rf_regression_scores, color="red", alpha=1)
    sns.histplot(xgb_regressionScores, color="green", alpha=1)
    sns.histplot(y_val, color="red", alpha=0.1)
    sns.histplot(y_train, color="yellow", alpha=0.25)
    plt.show()
    plt.clf()
