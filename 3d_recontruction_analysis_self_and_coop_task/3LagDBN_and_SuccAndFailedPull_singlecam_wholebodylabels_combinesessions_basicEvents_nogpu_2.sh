#!/bin/bash
#SBATCH --partition=pi_jadi
#SBATCH --job-name=3LagDBN_and_SuccAndFailedPull_singlecam_wholebodylabels_combinesessions_basicEvents_nogpu_2
#SBATCH --out="slurm-%j.out"
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weikang.shi@yale.edu

module purge
module load miniconda
conda activate DEEPLABCUT_YCRC
python 3LagDBN_and_SuccAndFailedPull_singlecam_wholebodylabels_combinesessions_basicEvents_2.py
