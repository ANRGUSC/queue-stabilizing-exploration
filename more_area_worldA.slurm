#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1GB
#SBATCH --time=6:00:00
#SBATCH --account=bkrishna_437

module load gcc/8.3.0
module load python/3.7.6
module load imagemagick

python3 lyapunov_explore.py more_area_world configA.json
./process_videos.bash
