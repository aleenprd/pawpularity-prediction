# %%
"""
# Regression Problem with tabular data
    # Use XG Boost
    # Use Neural Net 

# Unsupervised Clustering with K Means, DB Scan
    # Autoregressor popular data 
    # T SNE 

# Image Processing
    # GAN
    # Transfer Learning 
    # CNN Image Classification ¨(check gabriel preda with resnet 50)
    # Predict score from image itself 
    # Is it a cat or a dog?
    # What is the resolutin x by y of the photo
    # Does the photo include text 
    # Use the picture quality thingy neural net 
    # If possible, is the pet young or adult?
    # A lot of manual labelling going on 
    # Is the per de rasa? pedigree?
    # How many pets per photo?
    # Pet color
    # % of photo space (bound box) of pet 
    # coordinates of pet on photo
    # Study top 50 most popular pics
    # bottom fifty least popular pics 

https://www.kaggle.com/phylake1337/0-18-loss-simple-feature-extractors
https://github.com/idealo/image-quality-assessment
https://www.kaggle.com/yasufuminakama/petfinder-efficientnet-b0-lgb-inference

# How to label data into cats/dogs?
    # Are dogs more popular or?
"""
# %%
### IMPORT STANDARD PACKAGES
### ----------------- ####
import pickle
import pandas as pd 
import os 
import sys 
import numpy as np 
import math

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
from sklearn.metrics import mean_squared_error

# %%
### IMPORT CUSTOM PACKAGES
### ----------------- ####
currentDir = sys.path[0]
utilsPath = f"{currentDir}\\utils\\"
sys.path.append(utilsPath)

from utils import misc 
from utils import plots 
from utils import model

# %%
### RELOAD MODULES (remove after dev. stage)
### ----------------- ####
reload(misc)
reload(plots)
reload(model)

# %%
### DEFINE GLOBAL VARIABLES
### ----------------- ####
args = misc.argparser()
dataPath = f"{currentDir}\\data\\"
modelPath = f"{currentDir}\\model\\"

trainPath = f"{dataPath}\\train\\" # Image data 
testPath = f"{dataPath}\\test\\"
samplePath = f"{dataPath}\\test\\"
trainFileName = "train.csv"
testFileName = "test.csv"

plotsPath = f"{currentDir}\\plots\\"
edaPlotsPath = f"{plotsPath}\\eda\\"

# %%
### LOAD TABULAR DATA 
### ----------------- ####
train = pd.read_csv(f"{dataPath}\\{trainFileName}")
test = pd.read_csv(f"{dataPath}\\{testFileName}")

# %%
### EXPLORATORY DATA ANALYSIS 
### ----------------- ####
for df in [train,test]:
    misc.explore_data(df)
    plots.plot_all_donuts(df, colsToDrop=["Id","Pawpularity"], savePath=edaPlotsPath)

plots.make_correlation_matrix(train, dataName="Train Set", savePath=edaPlotsPath)

# %%
### REGRESSION PREDICTIVE ANALYSIS 
### ----------------- ####

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
# %%
# PREPROCESS DATA FOR REGRESSOR MODELS   
### ----------------- ####
X, y = model.regression_train_preprocess(train)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train Set Shape : {X_train.shape}")
print(f"Validation Set Shape : {X_val.shape}")

# %%
# TUNE, TRAIN AND PREDICT USING MODELS
### ----------------- ####
randomForestRegression.fit(X_train, y_train)
rfRegressionScores = randomForestRegression.predict(X_val)

xgbRegression.fit(X_train, y_train)
xgbRegressionScores = xgbRegression.predict(X_val)

# %%
# EVALUATE MODELS' PERFORMANCE ON VALIDATION SET
### ----------------- ####
print(np.std(train.Pawpularity)) # STDEV : 20.59
print(math.sqrt(mean_squared_error(y_val, rfRegressionScores))) # RMSE : 21.03
print(math.sqrt(mean_squared_error(y_val, xgbRegressionScores))) # RMSE : 21.00

# %%
# SAVE MODELS AS PICKLE FILES
### ----------------- ####
pickle.dump(randomForestRegression, open(f"{modelPath}\\randomForestRegression.pkl", 'wb'))
pickle.dump(xgbRegression, open(f"{modelPath}\\xgbRegression.pkl", 'wb'))

# %%
sns.histplot(rfRegressionScores)
sns.histplot(xgbRegressionScores)
sns.histplot(y_val)
sns.histplot(y_train)

# %%
def add_stars(df:pd.DataFrame)->pd.DataFrame:
    out = df['Pawpularity'].apply(lambda x: misc.popularity_to_stars(x))
    return out 

train['Stars'] = add_stars(train)
# %%
# Check the feature importance.
importance = pd.DataFrame({
    'features': features,
    'importance': xgb_model.feature_importances_
})
importance.sort_values(by='importance', inplace=True)

plt.barh([i for i in range(len(importance))], importance['importance'])
plt.title('XGBoost Feature Importance')
plt.show()
# %%
# Select informative features.
threshold = 0.005
importance = importance[importance['importance'] >= threshold]
plt.figure(figsize=(12, 16))
plt.barh(importance['features'], importance['importance'])
plt.title('XGBoost Feature Importance')
plt.savefig('features.png', dpi=300)
plt.show()