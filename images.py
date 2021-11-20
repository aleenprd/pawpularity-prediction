# %%
# IMPORT STANDARD PACKAGES

import pandas as pd
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm.notebook import tqdm
import warnings
from utils import misc
from utils import model
from utils import image as img


warnings.filterwarnings("ignore")


# %%
# DEFINE GLOBAL VARIABLES

dataPath, modelPath, trainPath, testPath, samplePath, trainFileName, \
    testFileName, plotsPath, edaPlotsPath = \
    model.define_global_variables()

cpuCount = mp.cpu_count()


def f_parallel(function, array):
    pool = mp.Pool(mp.cpu_count())
    out = list(
        tqdm(pool.imap(
            function, array),
            total=len(array))
            )
    pool.close()
    return out


imgTrainPath = "data/train/"
imgTestPath = "data/test"
trainSetImgIds = os.listdir(imgTrainPath)
testSetImgIds = os.listdir(imgTestPath)
print(trainSetImgIds[:5])

# LOAD TABULAR DATA

efficientDtype = misc.get_efficient_dtype()
trainTab = pd.read_csv(
    f"{dataPath}{trainFileName}", dtype=efficientDtype)
testTab = pd.read_csv(
    f"{dataPath}{testFileName}", dtype=efficientDtype)

# LOAD IMAGE DATA
for idx in trainSetImgIds[:5]:
    image = img.get_image(idx.rsplit(".", 1)[0], trainPath)
    plt.imshow(image)
    plt.show()
    plt.clf()

# %%
# MOST AND LEAST POPULAR IMAGES
mostPopularImages = trainTab.nlargest(20, 'Pawpularity')
leastPopularImages = trainTab.nsmallest(20, 'Pawpularity')

img.show_img_grid(mostPopularImages, imgTrainPath)
img.show_img_grid(leastPopularImages, imgTrainPath)

# %%
# FEATURE ENGINEERING
trainTab = trainTab.sort_values(by=['Id'], ascending=True)

# IMAGE DIMENSIONS
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

trainTab['Image Width'] = imageheightArray
trainTab['Imgage Height'] = imageWidthArray
trainTab['Dimension Ratio'] = imageDimensionRatioArray

trainTab.to_csv(f"{dataPath}processed/trainTab.csv", index=False)
