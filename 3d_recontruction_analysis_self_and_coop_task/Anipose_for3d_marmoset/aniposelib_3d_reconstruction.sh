#!/bin/bash
#SBATCH --partition=scavenge_gpu
#SBATCH --job-name=maniposelib_3d_reconstruction
#SBATCH --out="slurm-%j.out"
#SBATCH --ntasks=1 --nodes=1
#SBATCH --mem-per-cpu=5120
#SBATCH --time=0-24
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weikang.shi@yale.edu
#SBATCH --gpus=1

module purge
module load cuDNN
module load miniconda
module load Xvfb
conda activate DEEPLABCUT_YCRC
xvfb-run python aniposelib_3d_reconstruction.py
