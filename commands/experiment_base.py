import logging
import os
import subprocess
import time
import os

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

save_raw_data_base_dir = "/local_storage/users/amirme/raw_data/"
save_metadata_base_dir = "/local_storage/users/amirme/metadata/"


def set_logging_level(logging_level):
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    logger.setLevel(logging_level)


def _sweeper_cmd(
    **kwargs,
):
    (
        job_array_image_index,
        constrain,
        experiment_name,
        number_of_gpus,
    ) = handle_sbatch_args(kwargs)

    # handle method args
    method_args = " ".join([f"--{k} {v}" for k, v in kwargs.items()])
    method_args = method_args.replace("--demo False", "--no_demo")
    method_args = method_args.replace("--demo True", "")
    logger.debug(f"method_args: {method_args}")

    return (
        "sbatch "
        f"--job-name={experiment_name} "
        f"{constrain} "
        f"{job_array_image_index} "
        f"{number_of_gpus} "
        f"--export "
        f"method_args='"
        f"{method_args}"
        f"' "
        "commands/_sweeper.sbatch"
    )


def handle_sbatch_args(kwargs):
    job_array_image_index = ""
    constrain = ""
    experiment_name = "debug"
    number_of_gpus = ""

    if "experiment_name" in kwargs:
        experiment_name = kwargs["experiment_name"]
        logger.debug(f"--job-name={experiment_name}")
        del kwargs["experiment_name"]

    if "number_of_gpus" in kwargs:
        number_of_gpus = f"--gres=gpu:{kwargs['number_of_gpus']}"
        logger.debug(number_of_gpus)
        del kwargs["number_of_gpus"]

    if "job_array_image_index" in kwargs:
        job_array_image_index = f"--array={kwargs['job_array_image_index']}"
        logger.debug(f"job_array_image_index: {job_array_image_index}")
        del kwargs["job_array_image_index"]

    if "constraint" in kwargs:
        constrain = f"--constraint={kwargs['constraint']}"
        logger.debug(f"constraint: {constrain}")
        del kwargs["constraint"]

    return job_array_image_index, constrain, experiment_name, number_of_gpus


def run_experiment(**args):
    logger.debug("sumbitting a job")
    cmd = _sweeper_cmd(**args)
    os.system(cmd)
    wait_in_queue()


def wait_in_queue(thresh=10):
    while True:
        result = subprocess.run(["squeue", "-u", "amirme"], stdout=subprocess.PIPE)
        result = result.stdout.decode()
        result = len(result.split("\n")) - 2
        logger.debug(
            f"there are {result} number of jobs in the queue, waiting for finishing the jobs",
        )
        if result <= thresh:
            return
        time.sleep(5)
