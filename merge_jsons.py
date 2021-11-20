"""Script to merge some JSON files of the image quality scores."""


import os
import json


def merge_json_files(
    file_names: str,
    files_folder: str,
    merged_file_name: str
) -> None:
    """Merge several JSON files into one."""
    result = list()
    for f in file_names:
        with open(f"{files_folder}/{f}", 'r') as infile:
            result.extend(json.load(infile))
    with open(f"{files_folder}/{merged_file_name}", 'w') as output_file:
        json.dump(result, output_file)


if __name__ == "__main__":
    scores_folder = "data/image-quality"
    scores_file_names = os.listdir(scores_folder)

    merge_json_files(
        scores_file_names,
        scores_folder,
        "merged_json_img_quality_scores.json"
        )
