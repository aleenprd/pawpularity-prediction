#%%
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
import multiprocessing as mp
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
from utils import image as img

### RELOAD MODULES (remove after dev. stage)
### ----------------- ####
reload(misc)
reload(plots)
reload(model)
reload(img)

# %%
### DEFINE GLOBAL VARIABLES
### ----------------- ####

dataPath, modelPath, trainPath, testPath, samplePath, trainFileName, testFileName, plotsPath, edaPlotsPath = \
    model.define_global_variables()

cpuCount = mp.cpu_count()
def f_parallel(function, array):
    pool = mp.Pool(mp.cpu_count()) 
    out = list(tqdm(pool.imap(function, array),total = len(array)))
    pool.close()
    return out 

imgTrainPath = "data/train/"
imgTestPath = "data/test"
trainSetImgIds = os.listdir(imgTrainPath)
testSetImgIds = os.listdir(imgTestPath)
print(trainSetImgIds[:5])

### LOAD TABULAR DATA 
### ----------------- ####

efficientDtype = misc.get_efficient_dtype()
trainTab = pd.read_csv(f"{dataPath}\\{trainFileName}", dtype=efficientDtype)
testTab = pd.read_csv(f"{dataPath}\\{testFileName}", dtype=efficientDtype)

### LOAD IMAGE DATA 
### ----------------- ####

for idx in trainSetImgIds[:5]:
    image = img.get_image(idx.rsplit(".", 1)[0], trainPath)
    plt.imshow(image)
    plt.show()
    plt.clf()

# %%
### MOST AND LEAST POPULAR IMAGES
### ----------------- ####
mostPopularImages = trainTab.nlargest(20, 'Pawpularity')
leastPopularImages = trainTab.nsmallest(20, 'Pawpularity')

img.show_img_grid(mostPopularImages, imgTrainPath)
img.show_img_grid(leastPopularImages, imgTrainPath)

# %%
### FEATURE ENGINEERING
### ----------------- ####
trainTab = trainTab.sort_values(by=['Id'], ascending=True)

### IMAGE DIMENSIONS
### ----------------- ####
trainSetImgIds.sort(reverse=False)

trainSetFullPaths = []
for idx in tqdm(trainSetImgIds):
    trainSetFullPaths.append(f"{trainPath}/{idx}")

imageheightArray = []
imageWidthArray = []
imageDimensionRatioArray = []
for filePath in tqdm(trainSetFullPaths):
    dimensions = img.get_image_dimensions_from_path(filePath)
    imageheightArray.append(dimensions[0])
    imageWidthArray.append(dimensions[1])
    imageDimensionRatioArray.append(dimensions[2])
# imageDimensionRatiosArray = f_parallel(img.get_image_dimensions_from_path, trainSetFullPaths)

trainTab['Image Width'] = imageheightArray
trainTab['Imgage Height'] = imageWidthArray
trainTab['Dimension Ratio'] = imageDimensionRatioArray

trainTab.to_csv(f"{dataPath}/processed/trainTab.csv", index=False)

# %%
#bash ./predict --docker-image nima-cpu --base-model-name MobileNet --weights-file $(pwd)/models/MobileNet/weights_mobilenet_technical_0.11.hdf5 --image-source $(pwd)/src/tests/test_images
#bash .//predict --docker-image nima-cpu --base-model-name MobileNet --weights-file $(pwd)//models//MobileNet//weights_mobilenet_technical_0.11.hdf5 --image-source $(pwd)//src//tests//test_images
# bash .//predict --docker-image nima-cpu --base-model-name MobileNet --weights-file C://Users//Alin//Desktop//kaggle-paws//image-quality-assessment//models//MobileNet//weights_mobilenet_technical_0.11.hdf5 --image-source C://Users//Alin/Desktop//kaggle-paws//image-quality-assessment//src//tests//test_images