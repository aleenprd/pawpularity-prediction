import pickle
import pandas as pd 
import os 
import sys 
import numpy as np 
import math
from pathlib import Path

from importlib import reload 
from typing import List, Dict, Tuple
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns 

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def define_global_variables():
    currentDir = sys.path[0]

    dataPath = f"{currentDir}\\data\\"
    modelPath = f"{currentDir}\\model\\"

    trainPath = f"{dataPath}\\train\\" # Image data 
    testPath = f"{dataPath}\\test\\"
    samplePath = f"{dataPath}\\test\\"
    trainFileName = "train.csv"
    testFileName = "test.csv"

    plotsPath = f"{currentDir}\\plots\\"
    edaPlotsPath = f"{plotsPath}\\eda\\"

    return dataPath, modelPath, trainPath, testPath, samplePath, trainFileName, testFileName, plotsPath, edaPlotsPath

def regression_train_preprocess(df:pd.DataFrame)->List[pd.DataFrame]:
    df = df.set_index("Id")
    X = df.drop('Pawpularity', axis=1)
    y = df['Pawpularity']
    return X, y 

def build_rfr_pipeline():
    ### RANDOM FORESTS MODEL PIPELINE
    ### ----------------- ####
    randomGridRandomForests = {
        # Number of trees in random fores
        'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
        # Number of features to consider at every split
        'max_features': ['auto', 'sqrt'],
        # Maximum number of levels in tree
        'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
        # Minimum number of samples required to split a node
        'min_samples_split': [2, 5, 10],
        # Minimum number of samples required at each leaf node
        'min_samples_leaf': [1, 2, 3, 4],
        # Method of selecting samples for training each tree
        'bootstrap': [True, False]
                                } 
    print(randomGridRandomForests)
    print()

    randomForestRegression = Pipeline(
    [
        ('model', RandomizedSearchCV(
            estimator = RandomForestRegressor(), 
            param_distributions = randomGridRandomForests, 
            n_iter = 100, 
            cv = 3, 
            verbose=2, 
            random_state=42, 
            n_jobs = -1)
            )
    ]
    )
    return randomForestRegression

def build_xgbr_pipeline():
    ### XGBOOST MODEL PIPELINE
    ### ----------------- ####
    randomGridXgboost = {
        'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
        'learning_rate': [round(float(x),2) for x in np.linspace(start = 0.01, stop = 0.51, num = 10)],
        'subsample': [round(float(x),2) for x in np.linspace(start = 0.3, stop = 0.6, num = 4)],
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'colsample_bytree': [round(float(x),2) for x in np.linspace(start = 0.2, stop = 0.5, num = 4)],
        'min_child_weight': [1, 2, 3, 4]
                        }
    print(randomGridXgboost)
    print()

    xgbRegression = Pipeline(
    [
        ('model', RandomizedSearchCV(
            estimator = XGBRegressor(), 
            param_distributions = randomGridXgboost,
            cv = 5,  
            n_iter = 100, 
            scoring = 'neg_root_mean_squared_error', 
            error_score = 0, 
            verbose = 3, 
            n_jobs = -1)
            )
    ]
    )
    return xgbRegression

def print_score_statistics(df, dfName):
    if isinstance(df, pd.DataFrame):
        print(f"AVG {dfName} Set Score : {round(np.mean(df.Pawpularity),3)}") # AVG : 20.59
        print(f"STDEV. {dfName} Set Score : {round(np.std(df.Pawpularity),3)}") # STDEV : 20.59
    else:
        print(f"AVG {dfName} Set Score : {round(np.mean(df),3)}") # AVG : 20.59
        print(f"STDEV. {dfName} Set Score : {round(np.std(df),3)}") # STDEV : 20.59
    print()