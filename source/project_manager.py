import numpy as np
import pandas as pd
from glob import glob
import os

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def delete_experiment_data(metadata: pd.Series, selected_rows: pd.Series = None):
    assert isinstance(metadata, pd.Series)
    path = metadata.iloc[0]
    assert "amirme" in path, f"Path {path} does not contain amirme"
    assert path.endswith(".npy") or path.endswith(
        ".csv"
    ), f"Path {path} does not end with .npy or .csv"

    if selected_rows is None:
        return metadata.apply(os.remove)
    return metadata[selected_rows].apply(os.remove)


def check_file_exists(metadata: pd.Series, selected_rows: pd.Series = None):
    if selected_rows is None:
        return metadata.apply(os.path.exists)
    return metadata[selected_rows].apply(os.path.exists)


def load_experiment_metadata(save_metadata_dir, glob_path: str = "*.csv"):
    glob_path = os.path.join(save_metadata_dir, glob_path)
    metadata_paths = glob(glob_path)
    metadata_paths_merged = [path for path in metadata_paths if "merged" in path]
    assert (
        len(metadata_paths_merged) == 1
    ), f"Could not find any metadata files in {glob_path}"

    metadata_path = metadata_paths_merged[0]
    return pd.read_csv(metadata_path, index_col=False)


def load_experiment_inconsistency(save_metadata_dir, glob_path: str = "*.csv"):
    glob_path = os.path.join(save_metadata_dir, glob_path)
    metadata_paths = glob(glob_path)
    metadata_paths_inconsistency = [path for path in metadata_paths if "inconsistency" in path]
    assert (
        len(metadata_paths_inconsistency) == 1
    ), f"Could not find any metadata files in {glob_path}"

    metadata_path = metadata_paths_inconsistency[0]
    return pd.read_csv(metadata_path, index_col=False)


def merge_experiment_metadata(save_metadata_dir: str):
    glob_path: str = "*.csv"
    metadata_glob_path = os.path.join(save_metadata_dir, glob_path)
    metadata_paths = glob(metadata_glob_path)
    dataframes = []
    if metadata_paths:
        for metadata_path in metadata_paths:
            project_data = pd.read_csv(metadata_path)
            dataframes.append(project_data)
    else:
        raise FileNotFoundError(
            f"Could not find any metadata files in {metadata_glob_path}"
        )

    project_data = pd.concat(dataframes)
    save_metadata_path = os.path.join(save_metadata_dir, f"merged_metadata.csv")
    logger.info(f"Saving merged metadata to {save_metadata_path}")
    project_data.to_csv(save_metadata_path, index=False)
