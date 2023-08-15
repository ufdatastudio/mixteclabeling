#!/bin/bash
#SBATCH --job-name=mixteclabeling
#SBATCH --output=mixteclabeling.out
#SBATCH --error=mixteclabeling.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexwebber@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4

#module load ufrc
module load pytorch/2.0.1
#rm mixteclabeling.err mixteclabeling.out
#python3 -m pip install pytorch_lightning tensorboard pandas numpy matplotlib seaborn scikit-learn
# mamba install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

learning_rates=(0.1 0.01 0.001 0.0001)
batch_size=(64 128 256 512)
date_run=$(date +"%F")

# Double nested for loop
for learning_rate in "${learning_rates[@]}"; do
    for batch_size in "${batch_size[@]}"; do
        echo "Learning rate: $learning_rate"
        echo "Batch size: $batch_size"
        ## add date as argument
        python3 src/train.py --learning_rate $learning_rate --batch_size $batch_size --run "$date_run--lr$learning_rate--bs$batch_size--transformsHFlipVFlipRotationBlocks" &
    done
done

pwd
#cd /orange/ufdatastudios/christan/mixteclabeling/
python3 src/train.py
date