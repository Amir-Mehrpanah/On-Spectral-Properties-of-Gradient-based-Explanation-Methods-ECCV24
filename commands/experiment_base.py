import logging
import os
import subprocess
import time
import os

logger = logging.getLogger(__name__)


def set_logging_level(logging_level):
    logging.getLogger("source").setLevel(logging_level)
    logging.getLogger("commands").setLevel(logging_level)
    # logging.getLogger("source.utils").setLevel(logging_level)
    # logging.getLogger("source.driver_helpers").setLevel(logging_level)
    # logging.getLogger("commands.experiment_base").setLevel(logging_level)
    # logging.getLogger("source.explanation_methods.noise_interpolation").setLevel(
    #     logging_level
    # )
    logging.getLogger("__main__").setLevel(logging_level)
    logger.setLevel(logging_level)


def _sweeper_cmd(
    **kwargs,
):
    # handle slurm args
    job_array_image_index = ""
    constrain = ""
    if "job_array_image_index" in kwargs:
        job_array_image_index = f"--array={kwargs['job_array_image_index']}"
        logger.debug(f"job_array_image_index: {job_array_image_index}")
        del kwargs["job_array_image_index"]  # remove from kwargs
    if "constraint" in kwargs:
        constrain = f"--constraint={kwargs['constraint']}"
        logger.debug(f"constraint: {constrain}")
        del kwargs["constraint"]

    # handle method args
    method_args = " ".join([f"--{k} {v}" for k, v in kwargs.items()])
    method_args = method_args.replace("--demo False", "--no_demo")
    method_args = method_args.replace("--demo True", "")
    logger.debug(f"method_args: {method_args}")
    return (
        "sbatch "
        f"{constrain} "
        f"{job_array_image_index} "
        f"--export "
        f"method_args='"
        f"{method_args}"
        f"' "
        "commands/_sweeper.sbatch"
    )


def run_experiment(**args):
    logger.debug("sumbitting a job")
    cmd = _sweeper_cmd(**args)
    os.system(cmd)
    wait_in_queue()


def wait_in_queue(thresh=4):
    while True:
        result = subprocess.run(["squeue", "-u", "amirme"], stdout=subprocess.PIPE)
        result = result.stdout.decode()
        result = len(result.split("\n")) - 2
        logger.debug(
            f"there are {result} number of jobs in the queue, waiting for finishing the jobs",
        )
        if result <= thresh:
            break
        time.sleep(5)
