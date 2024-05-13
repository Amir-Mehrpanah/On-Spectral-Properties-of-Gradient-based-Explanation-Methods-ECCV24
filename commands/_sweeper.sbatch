#!/usr/bin/env bash
#SBATCH --time 08:00:00

date
echo cwd: $PWD
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi

module load Anaconda/2021.05-nsc1
conda activate xlab10
which python

#<array_process>
#MOD_PLACEHOLDER 
#</$array_process>

echo SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
scontrol show job $SLURM_JOB_ID
echo sbatch script: method args $method_args
echo sbatch script: array args $array_process

python driver.py --assert_device $array_process $method_args

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME} finished"
