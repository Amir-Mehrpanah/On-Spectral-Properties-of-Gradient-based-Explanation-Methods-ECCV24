import pandas as pd
import datetime
from glob import glob
import os


def save_slurm_api_data(df):
    file_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"outputs/slurm_api_data/{file_datetime}.csv", index=False)


def load_slurm_api_data(data_base_dir):
    file_datetime = glob("outputs/slurm_api_data/*.csv")[-1]
    if file_datetime:
        project_data = pd.read_csv(file_datetime)
    else:
        print("No slurm api data found creating data from paths")
        raise FileNotFoundError()

    data_path_isna = project_data["data_path"].isna()
    columns = project_data.columns
    columns.drop(["data_path"])
    project_data.loc[data_path_isna, "data_path"] = project_data.loc[
        data_path_isna, columns
    ].apply(path_from_project_data, axis=1)
    project_data["data_path"] = project_data["data_path"].apply(
        lambda x: os.path.join(data_base_dir, x)
    )
    return project_data


def noise_interpolation_args_to_str(args):
    return f"alpha={args['alpha']}"


arg_to_str_switch = {
    "noise_interpolation": noise_interpolation_args_to_str,
}


def args_to_str(method, args):
    method_arg_to_str = arg_to_str_switch[method]
    return method_arg_to_str(args)


def path_from_project_data(row):
    row["override_args"] = args_to_str(row["method"], row["override_args"])
    row = row.astype(str)
    file_path = "/".join(row)
    file_path = file_path
    return file_path
