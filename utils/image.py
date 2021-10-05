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

def get_image_file_path(imagesPath, imageId):
    return f'{imagesPath}/{imageId}.jpg'

def get_image_dimensions_from_path(pathToImage):
    image = cv2.imread(pathToImage)
    height, width, channels = image.shape
    ratio = height / width
    return (height, width, ratio)

def get_image_dimensions(image):
    height, width, channels = image.shape
    ratio = height / width
    return (height, width, ratio)

def get_image(idx, imageFolderPath):
    imagePath = os.path.join(f"{imageFolderPath}", idx+'.jpg'
        )
    return cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)

def show_image_by_id(idx, imageFolderPath):
    image = get_image(idx, imageFolderPath)
    plt.imshow(image)
    plt.show()
    plt.clf()

def show_img_grid(df, imageFolderPath, rows=4, cols=5):
    df = df.copy().reset_index()
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*4, rows*4))
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            image = get_image(df.loc[idx, "Id"], imageFolderPath)
            axes[r, c].imshow(image)
            axes[r, c].set_title(f'Top {idx}, Score {df.loc[idx, "Pawpularity"]}')