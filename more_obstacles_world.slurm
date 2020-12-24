#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=16:00:00
#SBATCH --account=bkrishna_437

module load gcc/8.3.0
module load python/3.7.6
module load imagemagick

python3 lyapunov_explore.py more_obstacles_world configC.json
