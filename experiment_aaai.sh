#!/bin/bash

learning_rates=(0.1 0.01 0.001 0.0001)
batch_sizes=(64 128 256 512)

# Double nested for loop
for learning_rate in "${learning_rates[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        ## add date as argument
        sleep 3
        sbatch runmodeltemplate.sh $learning_rate $batch_size
    done
done