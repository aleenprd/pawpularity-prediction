# %%
import os 
import json 

curDir = os.getcwd()
scoresFolder = f"{curDir}/data/image-quality"
scoresFileNames = os.listdir(scoresFolder)

def merge_json_files(fileNames, filesFolder, mergedFileName):
    result = list()
    for f in fileNames:
        with open(f"{filesFolder}\\{f}", 'r') as infile:
            result.extend(json.load(infile))
    with open(f"{filesFolder}\\{mergedFileName}", 'w') as outputFile:
        json.dump(result,outputFile)

merge_json_files(scoresFileNames, scoresFolder, f"merged_json_img_quality_scores.json")
