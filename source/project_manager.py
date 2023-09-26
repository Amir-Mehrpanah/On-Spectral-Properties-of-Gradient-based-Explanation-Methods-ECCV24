import pandas as pd
from glob import glob
import os

from source.configs import DefaultArgs


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


def load_experiment_metadata(glob_path: str = "*.csv", list_of_paths: list = None):
    if list_of_paths is None:
        metadata_glob_path = os.path.join(DefaultArgs.save_metadata_dir, glob_path)
        metadata_paths = glob(metadata_glob_path)
    else:
        metadata_paths = list_of_paths
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
    return project_data
