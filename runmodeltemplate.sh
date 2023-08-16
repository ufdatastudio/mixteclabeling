#!/bin/bash
#SBATCH --output=mixteclabeling-%x.%j.out
#SBATCH --error=mixteclabeling-%x.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexwebber@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5:00:00

#module load ufrc
module load pytorch/2.0.1

learning_rate=$1
batch_size=$2
date_run=$(date +"%F")

echo "Learning rate: $learning_rate"
echo "Batch size: $batch_size"
## add date as argument
python3 src/train.py --learning_rate $learning_rate --batch_size $batch_size --run "$date_run--lr$learning_rate--bs$batch_size--transformsNone"