import numpy as np
import pandas as pd
from glob import glob
import os

import tensorflow as tf


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


def merge_experiment_metadata(save_metadata_dir: str, path_prefix: str):
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
    save_metadata_path = os.path.join(save_metadata_dir, f"{path_prefix}_merged.csv")
    project_data.to_csv(save_metadata_path, index=False)


def alpha_group_loader(
    save_metadata_dir: str,
    batch_size: int,
    path_prefix: str,
    input_shape: tuple = None,
    prefetch_factor=4,
):
    merged_metadata_path = os.path.join(save_metadata_dir, f"{path_prefix}_merged.csv")
    if not os.path.exists(merged_metadata_path):
        raise FileNotFoundError(
            f"Could not find any merged metadata files in {merged_metadata_path}"
        )
    merged_metadata = pd.read_csv(merged_metadata_path)
    assert "data_path" in merged_metadata.columns, (
        f"Could not find data_path column in {merged_metadata_path}. "
        f"Make sure the metadata file contains a column named data_path"
    )
    if input_shape is None:
        assert "input_shape" in merged_metadata.columns, (
            f"Could not find input_shape column in {merged_metadata_path}. "
            f"Make sure the metadata file contains a column named input_shape"
            f"or pass input_shape as an argument to loader_from_metadata"
        )
        input_shape = merged_metadata["input_shape"].iloc[0]
        input_shape = tuple(input_shape)
    assert isinstance(
        input_shape, tuple
    ), f"input_shape must be a tuple, got {type(input_shape)}"
    assert (
        len(input_shape) == 3
    ), f"input_shape must have 3 dimensions (H, W, C), got {len(input_shape)}"

    assert "alpha_mask_value" in merged_metadata.columns, (
        f"Could not find alpha_mask_value column in {merged_metadata_path}. "
        f"Make sure the metadata file contains a column named alpha_mask_value"
    )

    num_alphas = merged_metadata["alpha_mask_value"].unique()
    input_shape = (num_alphas, *input_shape)

    def _generator():
        groupped = merged_metadata.groupby("alpha_mask_value")
        for i, paths in groupped:
            batch = paths["data_path"].apply(np.load)
            yield {
                "data": np.stack(batch.values),
                "indices": i,
            }

    dataset = tf.data.Dataset.from_generator(
        _generator,
        output_signature={"data": tf.TensorSpec(shape=input_shape, dtype=tf.float32)},
    )
    iterator = dataset.batch(batch_size).prefetch(prefetch_factor).as_numpy_iterator()
    return iterator
