import pandas as pd
import datetime
from glob import glob
import os

from source.configs import DefaultArgs


def delete_slurm_data(path):
    os.remove(path)


def check_slurm_data_exists(path):
    return os.path.exists(path)


def load_slurm_metadata():
    metadata_glob_path = os.path.join(DefaultArgs.save_metadata_dir, "*.csv")
    metadata_paths = glob(metadata_glob_path)
    dataframes = []
    if metadata_paths:
        for metadata_path in metadata_paths:
            project_data = pd.read_csv(metadata_path)
            dataframes.append(project_data)
    else:
        print("No slurm api data found creating data from paths")
        raise FileNotFoundError()

    project_data = pd.concat(dataframes)
    return project_data
