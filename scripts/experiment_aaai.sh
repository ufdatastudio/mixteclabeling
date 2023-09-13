#!/bin/bash

learning_rates=(0.0001 0.001 0.01)
batch_sizes=(128 64 32 16)

# Double nested for loop
for learning_rate in "${learning_rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        ## add date as argument
        sbatch -J "lr$learning_rate-bs$batch_size" runmodeltemplate.sh $learning_rate $batch_size
    done
done