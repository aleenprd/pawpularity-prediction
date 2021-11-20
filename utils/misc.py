"""Miscelaneous utilitarian functions."""


from IPython.display import display
import pandas as pd
import numpy as np
from typing import Dict


def explore_data(df: pd.DataFrame) -> None:
    """
    Explore dataframe in iPython, in terms of shape, data types, NaNs, etc.
    """
    display(df.shape)
    display(df.head())
    display(df.dtypes)
    display(df.isna().sum())
    display(df.describe())
    print()


def popularity_to_stars(x: int) -> int:
    """
    Turns a continuous popularity score into an ordinal star rating.
    """
    if x in range(0, 20):
        x = 1
    elif x in range(20, 40):
        x = 2
    elif x in range(40, 60):
        x = 3
    elif x in range(60, 80):
        x = 4
    elif x in range(80, 111):
        x = 5
    else:
        x = 0

    return x


def get_efficient_dtype() -> Dict:
    """Define a dataframe using efficient data types."""
    dataType = {
        'Id': 'string',
        'Subject Focus': np.uint8, 'Eyes': np.uint8, 'Face': np.uint8, 'Near': np.uint8,
        'Action': np.uint8, 'Accessory': np.uint8, 'Group': np.uint8, 'Collage': np.uint8,
        'Human': np.uint8, 'Occlusion': np.uint8, 'Info': np.uint8, 'Blur': np.uint8,
        'Pawpularity': np.uint8,
    }

    return dataType
