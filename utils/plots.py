import pandas as pd 
import numpy as np 
import os 
import sys 
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns 

def make_donut_chart(df:pd.DataFrame,colName:str,savePath=None)->plt.pie:
    # Create data
    sizeOfGroups = [x for x in df[colName].value_counts()]
    total = np.sum(sizeOfGroups)

    # Get names for labels 
    names = [x for x in df[colName].value_counts().keys()]
    percentages = []
    for i in range(0, len(names)):
            percentages.append(f"{names[i]} : {round(sizeOfGroups[i]/total*100,2):,}%")

    # Create a pieplot
    plt.pie(sizeOfGroups, labels=percentages)
    plt.suptitle(f"Distribution in {colName}", fontsize=16)

    # Add a circle at the center to transform it in a donut chart
    myCircle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(myCircle)

    # display and save
    if savePath:
        plt.savefig(f"{savePath}\\donut_chart_{colName}.jpeg")
    plt.show()
    plt.clf()

def plot_all_donuts(df:pd.DataFrame, colsToDrop:List[str], savePath=None)->plt.pie:
    # Get all colnames and filter them 
    colNames = list(df.columns)
    for col in colsToDrop:
        if col in colNames:
            colNames.remove(col)

    for colName in colNames:
        make_donut_chart(df, colName, savePath)

def make_correlation_matrix(df:pd.DataFrame, dataName:str, savePath=None,)->sns.heatmap:
    # get matrix 
    correlation_mat = df.corr()
    # define size
    plt.figure(figsize = (18,12))
    # draw heatmap
    ax = sns.heatmap(correlation_mat, annot = True, linewidths=.5)
    # decorate plot
    plt.title(f"Correlation Matrix {dataName}", fontsize=24)
    plt.xlabel("Image Tabular Features", fontsize=12)
    plt.ylabel("Image Tabular Features", fontsize=12)
    # display and save
    if savePath:
        plt.savefig(f"{savePath}\\corr_matrix_{dataName}.jpeg")
    plt.show()
    plt.clf()

def get_xgb_feature_importance(model, modelName, df, path):
    importance = pd.DataFrame({
        'features': df.columns,
        'importance': model.feature_importances_
    })
    importance.sort_values(by='importance', inplace=True)
    threshold = 0.005
    importance = importance[importance['importance'] >= threshold]
    plt.figure(figsize=(12, 16))
    plt.barh(importance['features'], importance['importance'])
    plt.title(f'{modelName} Feature Importance')
    plt.savefig(f"{path}\\{modelName}_features.png", dpi=300)
    plt.show()
    plt.clf()