import pandas as pd
from glob import glob
import os

import logging

logger = logging.getLogger(__name__)


def delete_experiment_data(metadata: pd.Series, selected_rows: pd.Series = None):
    assert isinstance(metadata, pd.Series)
    path = metadata.iloc[0]
    assert "x_amime" in path, f"Path {path} does not contain a username"
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
    ), f"Could not find any metadata files in {glob_path} found {metadata_paths_merged}"

    metadata_path = metadata_paths_merged[0]
    return pd.read_csv(metadata_path, index_col=False)


def load_experiment_inconsistency(save_metadata_dir, glob_path: str = "*.csv"):
    glob_path = os.path.join(save_metadata_dir, glob_path)
    metadata_paths = glob(glob_path)
    metadata_paths_inconsistency = [
        path for path in metadata_paths if "inconsistency" in path
    ]
    assert (
        len(metadata_paths_inconsistency) == 1
    ), f"Could not find any metadata files in {glob_path} found {metadata_paths_inconsistency}"

    metadata_path = metadata_paths_inconsistency[0]
    return pd.read_csv(metadata_path, index_col=False)


def merge_experiment_metadata(save_metadata_dir: str, glob_path: str = "*.csv"):
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


def compute_spectral_lens(save_metadata_dir, save_raw_data_dir):
    project_metadata = load_experiment_metadata(
        save_metadata_dir, glob_path="merged_*.csv"
    )
    logger.info(f"Loaded metadata from {save_metadata_dir}")
    project_metadata = (
        project_metadata.dropna()
        .set_index(
            [
                "stream_name",
                "stream_statistic",
                "image_index",
                "alpha_mask_value",
            ]
        )
        .sort_index()
    )
    explanations_temp = project_metadata.loc[
        ("vanilla_grad_mask", "meanx2", slice(None), slice(None)), "data_path"
    ]
    explanations_temp = explanations_temp.droplevel(["stream_name", "stream_statistic"])
    explanations_temp.name = "grad_mask"
    explanations_temp.sort_index(inplace=True)
    explanations_temp = explanations_temp.reset_index()

    from source.data_manager import save_spectral_lens

    explanations_mean_freq = explanations_temp.groupby(
        "image_index", as_index=True
    ).apply(
        save_spectral_lens,
        save_raw_data_dir=save_raw_data_dir,
    )
    explanations_mean_freq.name = "data_path"

    logger.debug(
        f"spectral lens shape before concatenating auxilary data {explanations_mean_freq.shape}"
    )
    # concat auxilary information
    f0 = project_metadata.index.get_level_values("alpha_mask_value")[0]
    explanations_temp = project_metadata.loc[
        ("vanilla_grad_mask", "meanx2", slice(None), f0), ["image_path", "label"]
    ]
    explanations_temp = explanations_temp.droplevel(
        ["stream_name", "stream_statistic", "alpha_mask_value"]
    )
    explanations_temp.sort_index(inplace=True)
    explanations_mean_freq = pd.concat(
        [explanations_mean_freq, explanations_temp], axis=1
    )
    explanations_mean_freq = explanations_mean_freq.reset_index()

    logger.debug(
        f"spectral lens shape after concatenating auxilary data {explanations_mean_freq.shape}"
    )

    explanations_temp = project_metadata.loc[
        ("vanilla_grad_mask", "meanx2", slice(None), slice(None)),
        ["data_path", "image_path", "label"],
    ]
    explanations_temp = explanations_temp.droplevel(["stream_name", "stream_statistic"])
    explanations_temp = explanations_temp.reset_index()
    logger.debug(
        f"merged_metadata shape before concatenating spectral lens metadata {explanations_temp.shape}"
    )

    explanations_temp = pd.concat([explanations_mean_freq, explanations_temp], axis=0)
    logger.debug(
        f"merged_metadata shape after concatenating spectral lens metadata {explanations_temp.shape}"
    )
    save_path = os.path.join(save_metadata_dir, "sl_merged_metadata.csv")
    explanations_temp.to_csv(save_path, index=False)
    logger.debug(f"saved sl_merged_metadata in {save_path}")