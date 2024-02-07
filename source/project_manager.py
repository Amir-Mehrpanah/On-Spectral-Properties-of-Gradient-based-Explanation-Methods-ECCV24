from functools import partial
import numpy as np
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


def compute_entropy(save_metadata_dir):
    metadata = load_experiment_metadata(
        save_metadata_dir, glob_path="merged_metadata.csv"
    )
    metadata = metadata.set_index(
        [
            "stream_name",
            "alpha_mask_value",
        ]
    )
    metadata = metadata.loc["log_probs", "data_path"]
    metadata = metadata.apply(lambda x: np.load(x))
    metadata = metadata.apply(lambda x: -(x * np.exp(x)).sum())
    metadata = metadata.groupby("alpha_mask_value").mean()
    metadata.name = "predictive_entropy"
    filename = os.path.join(save_metadata_dir, "predictive_entropy.csv")
    metadata = metadata.reset_index()
    metadata.to_csv(filename, index=False)


def merge_experiment_metadata(
    save_metadata_dir: str,
    glob_path: str = "*.csv",
    file_name: str = "merged_metadata.csv",
):
    metadata_glob_path = os.path.join(save_metadata_dir, glob_path)
    metadata_paths = glob(metadata_glob_path)
    dataframes = []
    if metadata_paths:
        for metadata_path in metadata_paths:
            if file_name in metadata_path:
                logger.debug(
                    f"Skipping {metadata_path}, will be overwritten by merge action"
                )
                continue
            project_data = pd.read_csv(metadata_path)
            dataframes.append(project_data)
    else:
        raise FileNotFoundError(
            f"Could not find any metadata files in {metadata_glob_path}"
        )

    project_data = pd.concat(dataframes)
    save_metadata_path = os.path.join(save_metadata_dir, file_name)
    logger.info(f"Saving merged metadata to {save_metadata_path}")
    project_data.to_csv(save_metadata_path, index=False)


def compute_with_explanation_prior(
    save_metadata_dir,
    save_raw_data_dir,
    stream_statistic,
    save_results_fn,
    alpha_mask_name,
    alpha_prior,
    ig_elementwise,
):
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
        ("vanilla_grad_mask", stream_statistic, slice(None), alpha_prior),
        ["data_path", "image_path"],
    ]

    explanations_temp = explanations_temp.droplevel(["stream_name", "stream_statistic"])
    explanations_temp = explanations_temp.rename(columns={"data_path": "grad_mask"})
    explanations_temp.sort_index(inplace=True)
    explanations_temp = explanations_temp.reset_index()

    logger.debug(
        f"statistics shape before computing the aggregation function {explanations_temp.shape}"
    )

    explanations_mean_freq = explanations_temp.groupby(
        "image_index", as_index=True
    ).apply(
        save_results_fn,
        save_raw_data_dir=save_raw_data_dir,
    )
    logger.debug(
        f"columns after computing the aggregation function {explanations_mean_freq.name}"
    )
    explanations_mean_freq.name = "data_path"

    logger.debug(
        f"statistics shape before concatenating auxilary data {explanations_mean_freq.shape}"
    )
    # concat auxilary information
    f0 = project_metadata.index.get_level_values("alpha_mask_value")[0]
    explanations_temp = project_metadata.loc[
        ("vanilla_grad_mask", stream_statistic, slice(None), f0),
        ["image_path", "label", "baseline_mask_type"],
    ]
    explanations_temp = explanations_temp.droplevel(
        ["stream_name", "stream_statistic", "alpha_mask_value"]
    )
    explanations_temp.sort_index(inplace=True)
    explanations_mean_freq = pd.concat(
        [explanations_mean_freq, explanations_temp], axis=1
    )
    explanations_mean_freq = explanations_mean_freq.reset_index()

    explanations_mean_freq["alpha_mask_value"] = alpha_mask_name
    explanations_mean_freq["stream_name"] = "vanilla_grad_mask"
    explanations_mean_freq["stream_statistic"] = stream_statistic
    explanations_mean_freq["ig_elementwise"] = ig_elementwise

    logger.debug(
        f"statistics shape after concatenating auxilary data {explanations_mean_freq.shape}"
    )

    save_path = os.path.join(save_metadata_dir, f"{alpha_mask_name}_metadata.csv")
    explanations_mean_freq.to_csv(save_path, index=False)
    logger.debug(f"saved {alpha_mask_name}_metadata in {save_path}")


def compute_integrated_grad(
    save_metadata_dir,
    save_raw_data_dir,
    stream_statistic="meanx2",
    alpha_mask_name="ig_sq",
    alpha_prior=slice(None),
    ig_elementwise=False,
):
    from source.data_manager import save_integrated_grad

    logger.debug(
        f"Computing integrated grad for stream_statistic {stream_statistic} "
        f"with alpha_mask_name {alpha_mask_name} and alpha_prior "
        f"{alpha_prior} with ig_elementwise {ig_elementwise}"
    )
    save_results_fn = partial(save_integrated_grad, ig_elementwise=ig_elementwise)
    compute_with_explanation_prior(
        save_metadata_dir,
        save_raw_data_dir,
        stream_statistic,
        save_results_fn=save_results_fn,
        alpha_mask_name=alpha_mask_name,
        alpha_prior=alpha_prior,
        ig_elementwise=ig_elementwise,
    )


def compute_spectral_lens(
    save_metadata_dir,
    save_raw_data_dir,
    stream_statistic="meanx2",
    alpha_mask_name="sl_sg",
    alpha_prior=slice(None),
):
    from source.data_manager import save_spectral_lens

    compute_with_explanation_prior(
        save_metadata_dir,
        save_raw_data_dir,
        stream_statistic,
        save_results_fn=save_spectral_lens,
        alpha_mask_name=alpha_mask_name,
        alpha_prior=alpha_prior,
        ig_elementwise=False,
    )
