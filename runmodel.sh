#!/bin/bash
#SBATCH --job-name=mixteclabeling
#SBATCH --output=mixteclabeling.out
#SBATCH --error=mixteclabeling.err
#SBATCH --mail-type=NONE
#SBATCH --mail-user=christan@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

#module load ufrc
module load pytorch/2.0.1
#rm mixteclabeling.err mixteclabeling.out
python3 -m pip install pytorch_lightning tensorboard
# mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pwd
#cd /orange/ufdatastudios/christan/mixteclabeling/
python3 src/train.py
date