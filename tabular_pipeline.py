# %%
### IMPORT STANDARD PACKAGES
### ----------------- ####
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
from sklearn.metrics import mean_squared_error, r2_score

import cv2
import os 
from multiprocessing import cpu_count
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore")

### IMPORT CUSTOM PACKAGES
### ----------------- ####
currentDir = sys.path[0]
utilsPath = f"{currentDir}\\utils\\"
sys.path.append(utilsPath)

from utils import misc 
from utils import plots 
from utils import model

### RELOAD MODULES (remove after dev. stage)
### ----------------- ####
reload(misc)
reload(plots)
reload(model)


# %%
### DEFINE GLOBAL VARIABLES
    ### ----------------- ####
dataPath, modelPath, trainPath, testPath, samplePath, trainFileName, testFileName, plotsPath, edaPlotsPath = \
    model.define_global_variables()

### LOAD TABULAR DATA 
### ----------------- ####

# Efficient Data Types
# REF: https://www.kaggle.com/markwijkhuizen/petfinder-eda-yolov5-obj-detection-tfrecords
dataType = {
    'Id': 'string',
    'Subject Focus': np.uint8, 'Eyes': np.uint8, 'Face': np.uint8, 'Near': np.uint8,
    'Action': np.uint8, 'Accessory': np.uint8, 'Group': np.uint8, 'Collage': np.uint8,
    'Human': np.uint8, 'Occlusion': np.uint8, 'Info': np.uint8, 'Blur': np.uint8,
    'Pawpularity': np.uint8,
} 

train = pd.read_csv(f"{dataPath}\\{trainFileName}", dtype=dataType)
test = pd.read_csv(f"{dataPath}\\{testFileName}", dtype=dataType)

### EXPLORATORY DATA ANALYSIS 
### ----------------- ####
for df in [train,test]:
    misc.explore_data(df)
    plots.plot_all_donuts(df, colsToDrop=["Id","Pawpularity"], savePath=edaPlotsPath)

plots.make_correlation_matrix(train, dataName="Train Set", savePath=edaPlotsPath)


# %%
### REGRESSION PREDICTIVE ANALYSIS 
### ----------------- ####

# PREPROCESS DATA FOR REGRESSOR MODELS   
### ----------------- ####
X, y = model.regression_train_preprocess(train)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train Set Shape : {X_train.shape}")
print(f"Validation Set Shape : {X_val.shape}")


### LOAD MODELS (OPTIONAL : TUNE, TRAIN, SAVE)
### ----------------- ####
if Path(f"{modelPath}\\randomForestRegression.pkl").is_file():
    randomForestRegression = pickle.load(open(f"{modelPath}\\randomForestRegression.pkl", 'rb'))
    
else: 
    randomForestRegression = model.build_rfr_pipeline().fit(X_train, y_train)
    
    pickle.dump(randomForestRegression, open(f"{modelPath}\\randomForestRegression.pkl", 'wb'))

if Path(f"{modelPath}\\xgbRegression.pkl").is_file():
    xgbRegression = pickle.load(open(f"{modelPath}\\xgbRegression.pkl", 'rb'))
    
else:
    xgbRegression = model.build_xgbr_pipeline().fit(X_train, y_train)
    pickle.dump(xgbRegression, open(f"{modelPath}\\xgbRegression.pkl", 'wb'))


# %%
# EVALUATE MODELS' PERFORMANCE ON VALIDATION SET
### ----------------- ####

# These are the most important features the models consider for
# splitting the data and building leaves and branches
xgbModelRandCv = xgbRegression['model']
xgbModel = xgbModelRandCv.best_estimator_
plots.get_xgb_feature_importance(xgbModel, "XGBoost", X_train, edaPlotsPath)

rfModelRandCv = randomForestRegression['model']
rfModel = rfModelRandCv.best_estimator_
plots.get_xgb_feature_importance(rfModel, "RandomForests", X_train, edaPlotsPath)

# Our models are stupid and are basically almost as if predicting the mean all the time 
# We really need to include data on the images as engineered features
# A Neural Network could probably learn better than these methods, but still...

y_naiveScores = [np.mean(y_val)] * len(y_val + 1)
rfRegressionScores = randomForestRegression.predict(X_val)
xgbRegressionScores = xgbRegression.predict(X_val)

model.print_score_statistics(y, "Full Dataset")
model.print_score_statistics(y_train, "Train")
model.print_score_statistics(y_val, "Validation")

model.print_score_statistics(y_naiveScores, "Naive Method")
model.print_score_statistics(pd.Series(rfRegressionScores), "RFR Scores")
model.print_score_statistics(xgbRegressionScores, "XGB Scores")

# RMSE is the King metric in this contest and here's how our models compare
# to the 'naive' baseline 
print(f"RMSE Naive Method : {round(math.sqrt(mean_squared_error(y_val, y_naiveScores)),3)}") #21.025
print(f"RMSE Random Forests : {round(math.sqrt(mean_squared_error(y_val, rfRegressionScores)),3)}") #21.03
print(f"RMSE XGBoostt : {round(math.sqrt(mean_squared_error(y_val, xgbRegressionScores)),3)}") #21.00
print()
# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
# A constant model that always predicts the expected value of y, disregarding the input features, 
# would get a  score of 0.0.
print(f"R2 Naive Method : {round(r2_score(y_val, y_naiveScores),3)}") #0.0
print(f"R2 Random Forests : {round(r2_score(y_val, rfRegressionScores),3)}") #-0.001
print(f"R2 XGBoostt : {round(r2_score(y_val, xgbRegressionScores),3)}") #0.002

# The distributions in the predicted
# scores are mostly around the mean  
plt.figure(figsize=(16, 12))
plt.title('Red: RFR Scores vs Green: XGB Scores')
sns.histplot(rfRegressionScores, color="red", alpha=0.25)
sns.histplot(xgbRegressionScores, color="green", alpha=0.5)
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
sns.histplot(rfRegressionScores, color="red", alpha=1)
sns.histplot(xgbRegressionScores, color="green", alpha=1)
sns.histplot(y_val, color="red", alpha=0.1)
sns.histplot(y_train, color="yellow", alpha=0.25)
plt.show()
plt.clf()
