import logging
import os
import subprocess
import time
import os


def _sweeper_cmd(
    job_array_image_index,
    **kwargs,
):
    method_args = " ".join([f"--{k} {v}" for k, v in kwargs.items()])

    method_args = method_args.replace("--demo False", "--no_demo")
    method_args = method_args.replace("--demo True", "")
    logging.log(logging.DEBUG, f"method_args: {method_args}")
    return (
        "sbatch --constraint=gondor "
        f"--array={job_array_image_index} --export "
        f"method_args='"
        f"{method_args}"
        f"' "
        "commands/_sweeper.sbatch"
    )


def run_experiment(job_array_image_index, **args):
    logging.log(logging.DEBUG, "sumbitting a job")
    cmd = _sweeper_cmd(job_array_image_index, **args)
    os.system(cmd)
    _wait_in_queue()


def _wait_in_queue():
    result = 10
    while result > 6:
        result = subprocess.run(["squeue", "-u", "amirme"], stdout=subprocess.PIPE)
        result = result.stdout.decode()
        result = len(result.split("\n")) - 2
        logging.log(
            logging.DEBUG,
            f"there are {result} number of jobs in the queue, waiting for finishing the jobs",
        )
        time.sleep(5)
