from glob import glob
import logging
import os
import subprocess
import sys
import time
import os
from typing import List

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

cwd = os.getcwd()
assert cwd.endswith(
    "an_explanation_model"
), f"please run this script from the root of the project cwd: {cwd}"

username = "x_amime"
save_raw_data_base_dir = os.path.join(cwd, "outputs/raw_data/")
save_metadata_base_dir = os.path.join(cwd, "outputs/metadata/")
save_temp_base_dir = os.path.join(cwd, "outputs/temp/")
slurm_logs_base_dir = os.path.join(cwd, "outputs/slurm_logs/")

os.makedirs(save_raw_data_base_dir, exist_ok=True)
os.makedirs(save_metadata_base_dir, exist_ok=True)
os.makedirs(slurm_logs_base_dir, exist_ok=True)
os.makedirs(save_temp_base_dir, exist_ok=True)

def remove_files(base_path, glob_path="*", exclude="merged"):
    files = glob(os.path.join(base_path, glob_path))
    for f in files:
        if exclude in f:
            continue
        os.remove(f)


def set_logging_level(logging_level):
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    logger.setLevel(logging_level)


def _sweeper_cmd(
    **kwargs,
):
    experiment_name, slurm_args, array_process, sweeper_name = handle_sbatch_args(
        kwargs
    )
    sweeper_tmp = create_temp_sweeper_file(
        experiment_name=experiment_name,
        array_process=array_process,
        sweeper_name=sweeper_name,
    )

    # handle method args
    method_args = " ".join([f"--{k} {v}" for k, v in kwargs.items()])
    method_args = method_args.replace("--demo False", "--no_demo")
    method_args = method_args.replace("--demo True", "")

    logger.debug(f"method_args: {method_args}")
    logger.debug(f"array_process: {array_process}")

    return (
        "sbatch "
        f"{slurm_args} "
        f"--export "
        f"method_args='"
        f"{method_args}"
        f"' "
        f"{sweeper_tmp} "
    )


def create_temp_sweeper_file(experiment_name, array_process, sweeper_name):
    os.makedirs("commands/temp", exist_ok=True)
    # load _sweeper.sbatch
    with open(f"commands/{sweeper_name}", "r") as f:
        sweeper = f.read()
    # replace #MOD_PLACEHOLDER with array_process
    sweeper = sweeper.replace("#MOD_PLACEHOLDER", array_process)
    # write to file_name
    file_name = f"commands/temp/_sweeper_mod_{experiment_name}.sbatch"
    with open(file_name, "w") as f:
        f.write(sweeper)
        logger.debug(f"created {file_name}")

    return file_name


def handle_sbatch_args(kwargs):
    sweeper_name = "_sweeper.sbatch"
    array_process = ""
    job_array = ""
    constraint = ""
    experiment_name_cmd = "debug"
    number_of_gpus = ""
    if "experiment_name" in kwargs:
        experiment_name = kwargs["experiment_name"]
        experiment_name_cmd = f"--job-name={experiment_name} "
        logger.debug(f"experiment name is set to {experiment_name}")
        del kwargs["experiment_name"]

    if "number_of_gpus" in kwargs:
        number_of_gpus = f"--gpus={kwargs['number_of_gpus']} "
        logger.debug(f"number of gpus in handling slurm args {number_of_gpus}")
        del kwargs["number_of_gpus"]

    if "job_array" in kwargs:
        job_array = f"--array={kwargs['job_array']} "
        logger.debug(f"job_array: {job_array}")
        del kwargs["job_array"]

    if "constraint" in kwargs:
        constraint = f"--constraint={kwargs['constraint']} "
        logger.debug(f"constraint: {constraint}")
        del kwargs["constraint"]

    if "array_process" in kwargs:
        array_process = f"{kwargs['array_process']} "
        logger.debug(f"array_process: {array_process}")
        del kwargs["array_process"]

    if "sweeper_name" in kwargs:
        sweeper_name = kwargs["sweeper_name"]
        logger.debug(f"sweeper_name: {sweeper_name}")
        del kwargs["sweeper_name"]
    slurm_logs = os.path.join(slurm_logs_base_dir, f"%J_%x_%N.out")
    slurm_errs = os.path.join(slurm_logs_base_dir, f"%J_%x_%N.err")
    slurm_args = (
        f"--output={slurm_logs} "
        f"--error={slurm_errs} "
        f"{experiment_name_cmd}"
        f"{number_of_gpus}"
        f"{job_array}"
        f"{constraint}"
    )

    return experiment_name, slurm_args, array_process, sweeper_name


def run_experiment(**args):
    logger.debug("sumbitting a job")
    cmd = _sweeper_cmd(**args)
    os.system(cmd)
    # wait_in_queue()


def wait_in_queue(thresh=50, jobnames: List[str] = None):
    command = ["squeue", "-u", username]
    if jobnames is not None:
        command.append("--name")
        if isinstance(jobnames, list):
            jobnames = ",".join(jobnames)
        command.append(jobnames)

    while True:
        result = subprocess.run(command, stdout=subprocess.PIPE)
        result = result.stdout.decode()
        result = len(result.split("\n")) - 2
        logger.debug(
            f"there are {result} number of jobs in the queue, waiting for finishing the jobs",
        )
        if result <= thresh:
            return
        time.sleep(5)
