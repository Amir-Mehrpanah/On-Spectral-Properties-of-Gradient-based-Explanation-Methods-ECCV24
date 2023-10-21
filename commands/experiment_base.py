import logging
import os
import subprocess
import time
import os


def _sweeper_cmd(
    **kwargs,
):
    # handle slurm args
    job_array_image_index = ""
    constrain = ""
    if "job_array_image_index" in kwargs:
        job_array_image_index = f"--array={kwargs['job_array_image_index']}"
        logging.log(logging.DEBUG, f"job_array_image_index: {job_array_image_index}")
        del kwargs["job_array_image_index"]  # remove from kwargs
    if "constraint" in kwargs:
        constrain = f"--constraint={kwargs['constraint']}"
        logging.log(logging.DEBUG, f"constraint: {constrain}")
        del kwargs["constraint"]


    # handle method args
    method_args = " ".join([f"--{k} {v}" for k, v in kwargs.items()])
    method_args = method_args.replace("--demo False", "--no_demo")
    method_args = method_args.replace("--demo True", "")
    logging.log(logging.DEBUG, f"method_args: {method_args}")
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
    logging.log(logging.DEBUG, "sumbitting a job")
    cmd = _sweeper_cmd(**args)
    os.system(cmd)
    _wait_in_queue()


def _wait_in_queue(thresh=4):
    while True:
        result = subprocess.run(["squeue", "-u", "amirme"], stdout=subprocess.PIPE)
        result = result.stdout.decode()
        result = len(result.split("\n")) - 2
        logging.log(
            logging.DEBUG,
            f"there are {result} number of jobs in the queue, waiting for finishing the jobs",
        )
        if result < thresh:
            break
        time.sleep(5)
