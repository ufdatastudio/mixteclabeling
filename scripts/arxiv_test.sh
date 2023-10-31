#!/bin/bash
#SBATCH --output=out/logs/mixteclabeling-%x.%j.out
#SBATCH --error=out/logs/err/mixteclabeling-%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexwebber@ufl.edu
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00

#module load ufrc
module load pytorch/2.0.1

## add date as argument
python3 src/train.py --learning_rate 0.00025 --batch_size 64 --model "vgg16" --transforms "None" --category "gender"