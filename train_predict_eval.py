# %%
### IMPORT STANDARD PACKAGES
### ----------------- ####

import os 
import sys 
import math
import numpy as np 
from pathlib import Path
from multiprocessing import cpu_count
from tqdm.notebook import tqdm
import pandas as pd 
import pickle
import json
import seaborn as sns 
import matplotlib.pyplot as plt
from importlib import reload 
from typing import List, Dict, Tuple
from argparse import ArgumentParser
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
currentDir = sys.path[0]
dataPath = f"{currentDir}\\data\\"
modelPath = f"{currentDir}\\model\\"
processedPath = f"{dataPath}\\processed\\" 
plotsPath = f"{currentDir}\\plots\\"
edaPlotsPath = f"{plotsPath}\\eda\\"

trainFileName = "trainTab.csv"
testFileName = "test.csv"
catClassFileName = "cat_class.csv"
scoresFileName = "aesth_scores.json"

trainFullPath = os.path.join(processedPath,trainFileName)
testFullPath = os.path.join(processedPath,testFileName)
classFullPath =os.path.join(processedPath,catClassFileName)
scoresFullPath =os.path.join(processedPath,scoresFileName)

### LOAD TABULAR DATA 
### ----------------- ####
train = pd.read_csv(trainFullPath)
test = pd.read_csv(testFullPath)
cat = pd.read_csv(classFullPath)
with open(scoresFullPath, 'r') as dataFile:
    data = json.load(dataFile)
aesth = pd.DataFrame(data)

# %%
### INSPECT AND MERGE DATA 
### ----------------- ####
display(train.head())
display(test.head())
display(cat.head()) # https://www.kaggle.com/chrisbradley/code
display(aesth.head())

train['Is Cat'] = cat['is_cat']
train['Aesth Score'] = aesth['mean_score_prediction']
display(train)

# %%
### EXPLORATORY DATA ANALYSIS 
### ----------------- ####
for df in [train,test]:
    misc.explore_data(df)
    plots.plot_all_donuts(df, colsToDrop=["Id","Pawpularity"], savePath=edaPlotsPath)

# %%
plots.make_correlation_matrix(train, dataName="Train Set", savePath=edaPlotsPath)

# %%
# PREPROCESS DATA FOR REGRESSOR MODELS   
### ----------------- ####
X, y = model.regression_train_preprocess(train)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train Set Shape : {X_train.shape}")
print(f"Validation Set Shape : {X_val.shape}")

# %%
### LOAD MODELS (OPTIONAL : TUNE, TRAIN, SAVE)
### ----------------- ####

randomForestRegression = model.build_rfr_pipeline().fit(X_train, y_train)
pickle.dump(randomForestRegression, open(f"{modelPath}\\randomForestRegression3.pkl", 'wb'))

xgbRegression = model.build_xgbr_pipeline().fit(X_train, y_train)
pickle.dump(xgbRegression, open(f"{modelPath}\\xgbRegression3.pkl", 'wb'))

# %%
# EVALUATE MODELS' PERFORMANCE ON VALIDATION SET
### ----------------- ####

# These are the most important features the models consider for
# splitting the data and building leaves and branches 
rfModelRandCv = randomForestRegression['model']
rfModel = rfModelRandCv.best_estimator_
plots.get_xgb_feature_importance(rfModel, "RandomForests", X_train, edaPlotsPath)

xgbModelRandCv = xgbRegression['model']
xgbModel = xgbModelRandCv.best_estimator_
plots.get_xgb_feature_importance(xgbModel, "XGBoost", X_train, edaPlotsPath)

# %%
y_naiveScores = [np.mean(y_val)] * len(y_val + 1)
rfRegressionScores = randomForestRegression.predict(X_val)
xgbRegressionScores = xgbRegression.predict(X_val)

model.print_score_statistics(y, "Full Dataset")
model.print_score_statistics(y_train, "Train")
model.print_score_statistics(y_val, "Validation")

model.print_score_statistics(y_naiveScores, "Naive Method")
model.print_score_statistics(pd.Series(rfRegressionScores), "RFR Scores")
model.print_score_statistics(xgbRegressionScores, "XGB Scores")

# %%
# RMSE is the King metric in this contest and here's how our models compare
# to the 'naive' baseline 
print(f"RMSE Naive Method : {round(math.sqrt(mean_squared_error(y_val, y_naiveScores)),3)}") # 21.025 
print(f"RMSE Random Forests : {round(math.sqrt(mean_squared_error(y_val, rfRegressionScores)),3)}") # 20.576 (cats) # 20.558 (aesth)
print(f"RMSE XGBoostt : {round(math.sqrt(mean_squared_error(y_val, xgbRegressionScores)),3)}") # 20.524 # 20.518
print()

# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
# A constant model that always predicts the expected value of y, disregarding the input features, 
# would get a  score of 0.0.
print(f"R2 Naive Method : {round(r2_score(y_val, y_naiveScores),3)}") # 0.0 # 0.0 
print(f"R2 Random Forests : {round(r2_score(y_val, rfRegressionScores),3)}") # 0.042 (iscat and dims) # 0.044 (aesth)
print(f"R2 XGBoostt : {round(r2_score(y_val, xgbRegressionScores),3)}") # 0.047 # 0.048

# %%
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names = boston.feature_names[sorted_feature_importance])