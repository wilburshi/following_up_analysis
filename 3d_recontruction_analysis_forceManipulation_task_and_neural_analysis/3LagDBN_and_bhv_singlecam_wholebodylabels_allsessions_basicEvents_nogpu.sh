#!/bin/bash
#SBATCH --partition=week
#SBATCH --job-name=3LagDBN_and_bhv_singlecam_wholebodylabels_allsessions_basicEvents_nogpu
#SBATCH --out="slurm-%j.out"
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=60000
#SBATCH --time=144:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weikang.shi@yale.edu

module purge
module load miniconda
conda activate DLC
python 3LagDBN_and_bhv_singlecam_wholebodylabels_allsessions_basicEvents.py
