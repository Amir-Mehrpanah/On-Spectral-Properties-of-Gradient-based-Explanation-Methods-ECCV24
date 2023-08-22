import pandas as pd
import datetime
from glob import glob
import os


def create_slurm_api_data_from_paths(search_paths):
    """
    Creates a dataframe from paths that created after running slurm jobs
    implement this if you have accidentally deleted the slurm api data
    Note that there is a one to one mapping between the slurm api data and the
    paths that are created after running the slurm jobs.
    """
    raise NotImplementedError("Implement this if you have accidentally deleted the slurm api data")


def save_slurm_api_data(df):
    file_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"outputs/slurm_api_data/{file_datetime}.csv", index=False)


def load_slurm_api_data(data_base_dir):
    file_datetime = glob("outputs/slurm_api_data/*.csv")[-1]
    if file_datetime:
        project_data = pd.read_csv(file_datetime)
    else:
        print("No slurm api data found creating data from paths")
        project_data = create_slurm_api_data_from_paths(data_base_dir)

    data_path_isna = project_data["data_path"].isna()
    columns = project_data.columns
    columns.drop(["data_path", "data_exists"])
    project_data.loc[data_path_isna, "data_path"] = project_data.loc[
        data_path_isna, columns
    ].apply(path_from_project_data, axis=1)
    project_data["data_path"] = project_data["data_path"].apply(
        lambda x: os.path.join(data_base_dir, x)
    )
    project_data["data_exists"] = project_data.apply(
        lambda x: os.path.exists(x["data_path"]), axis=1
    )
    return project_data


def noise_interpolation_args_to_str(args):
    return f"alpha={args['alpha']}"


methods_arg_to_str = {
    "noise_interpolation": noise_interpolation_args_to_str,
}


def path_from_project_data(row, file_extention=".npy"):
    row["override_args"] = methods_arg_to_str[row["method"]](row["override_args"])
    row = row.astype(str)
    file_path = "/".join(row)
    file_path = file_path + file_extention
    return file_path
