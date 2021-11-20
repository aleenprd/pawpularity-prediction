"""Utils for dealing with images."""

import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import warnings
from typing import List


warnings.filterwarnings("ignore")


def get_image_file_path(images_path: str, image_id: str) -> str:
    """
    Get the path of an image.

    Obs: os.path exists. This is obsolete.
    """
    return f'{images_path}/{image_id}.jpg'


def get_image_dimensions(image: cv2) -> List:
    """Get height, width and ratio from image cv2 object."""
    height, width, channels = image.shape
    ratio = height / width

    return [height, width, ratio]


def get_image_dimensions_from_path(path_to_image: str) -> List:
    """Get height, width and ratio from image path."""
    image = cv2.imread(path_to_image)
    height, width, channels = image.shape
    ratio = height / width

    return [height, width, ratio]


def get_image(idx: int, image_folder_path: str) -> cv2:
    """Load image from path."""
    image_path = os.path.join(f"{image_folder_path}", idx+'.jpg')

    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def show_image_by_id(idx: int, image_folder_path: str):
    """Show image on screen, based on its id."""
    image = get_image(idx, image_folder_path)
    plt.imshow(image)
    plt.show()
    plt.clf()


def show_img_grid(df: pd.DataFrame, image_folder_path: str, rows=4, cols=5):
    """Show a grid of several images."""
    df = df.copy().reset_index()
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*4, rows*4))
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            image = get_image(df.loc[idx, "Id"], image_folder_path)
            axes[r, c].imshow(image)
            axes[r, c].set_title(
                f'Top {idx}, Score {df.loc[idx, "Pawpularity"]}')
