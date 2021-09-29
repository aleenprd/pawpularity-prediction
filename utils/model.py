import pandas as pd 
from typing import List
from xgboost import XGBRegressor

def regression_train_preprocess(df:pd.DataFrame)->List[pd.DataFrame]:
    df = df.set_index("Id")
    X = df.drop('Pawpularity', axis=1)
    y = df['Pawpularity']
    return X, y 