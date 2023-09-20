#!/bin/bash

learning_rates=(0.0001 0.00025)
batch_sizes=(128 64 32 16)
model="vit_l_16"

# Double nested for loop
for learning_rate in "${learning_rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        ## add date as argument
        sbatch -J "lr$learning_rate-bs$batch_size" scripts/runmodeltemplate.sh $learning_rate $batch_size $model
    done
done