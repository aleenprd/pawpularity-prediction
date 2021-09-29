import pandas as pd 
import numpy as np 
import os 
import sys 
from importlib import reload 
from typing import List, Dict, Tuple
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns 

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

def popularity_to_stars(x):
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
