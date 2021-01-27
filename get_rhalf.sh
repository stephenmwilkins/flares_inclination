#!/bin/bash
#SBATCH --ntasks 1
#SBATCH -A dp004
#SBATCH -p cosma7
#SBATCH --job-name=get_rhalf
#SBATCH -t 0-00:30
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-39%10
#SBATCH -o logs/std_output.%J
#SBATCH -e logs/std_error.%J

module purge
module load gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3 python/3.6.5


#export PY_INSTALL=/cosma/home/dp004/dc-love2/.conda/envs/eagle/bin/python

source ../flares_pipeline/venv_fl/bin/activate

python3 r_half.py $SLURM_ARRAY_TASK_ID 009_z006p000


echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit
