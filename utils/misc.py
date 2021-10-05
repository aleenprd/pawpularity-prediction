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
from sklearn.metrics import mean_squared_error

### IMPORT CUSTOM PACKAGES
### ----------------- ####
currentDir = sys.path[0]
utilsPath = f"{currentDir}\\utils\\"
sys.path.append(utilsPath)

from utils import plots 
from utils import model


def argparser():
    args = ArgumentParser()
    return args 

def explore_data(df):
    display(df.shape)
    display(df.head())
    display(df.dtypes)
    display(df.isna().sum())
    display(df.describe())
    print()

def popularity_to_stars(x:int)->int:
    if x in range(0,20):
        x = 1
    elif x in range(20,40):
        x = 2
    elif x in range(40,60):
        x = 3
    elif x in range(60,80):
        x = 4
    elif x in range(80,111):
        x = 5
    else:
        x = 0

def get_efficient_dtype():
    dataType = {
        'Id': 'string',
        'Subject Focus': np.uint8, 'Eyes': np.uint8, 'Face': np.uint8, 'Near': np.uint8,
        'Action': np.uint8, 'Accessory': np.uint8, 'Group': np.uint8, 'Collage': np.uint8,
        'Human': np.uint8, 'Occlusion': np.uint8, 'Info': np.uint8, 'Blur': np.uint8,
        'Pawpularity': np.uint8,
    } 
    return dataType
"""
def add_stars(df:pd.DataFrame)->pd.DataFrame:
    out = df['Pawpularity'].apply(lambda x: misc.popularity_to_stars(x))
    return out 

train['Stars'] = add_stars(train)
"""