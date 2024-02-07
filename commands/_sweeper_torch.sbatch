#!/usr/bin/env bash
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 4G
#SBATCH --time 12:00:00

#<array_process>
#MOD_PLACEHOLDER 
#</$array_process>

date
echo cwd: $PWD
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi

module load Anaconda/2021.05-nsc1
conda activate torch8
which python

echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
echo sbatch script: method args $method_args
echo sbatch script: array args $array_process

python source/torch_driver.py --assert_device $array_process $method_args

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} finished"
