#!/bin/bash
#SBATCH --job-name=mixteclabeling
#SBATCH --output=mixteclabeling.out
#SBATCH --error=mixteclabeling.err
#SBATCH --mail-type=NONE
#SBATCH --mail-user=christan@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

#module load ufrc
module load pytorch
#rm mixteclabeling.err mixteclabeling.out

mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pwd
#cd /orange/ufdatastudios/christan/mixteclabeling/
python3 src/train.py
