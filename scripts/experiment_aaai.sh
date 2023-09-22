#!/bin/bash

learning_rates=(0.0005 0.001 0.01 0.1)
batch_sizes=(128 64 32 16)
model=("vit_l_16" "vgg16")
transforms=("RandomErasing_RandomHorizontalFlip_RandomVerticalFlip" "RandomErasing" "RandomHorizontalFlip_RandomVerticalFlip" "None")

# create an array with all variables above


# Double nested for loop
for model in "${model[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            for transform in "${transforms[@]}"; do
                sbatch -J "lr$learning_rate-bs$batch_size-model$model-transforms$transform" scripts/runmodeltemplate.sh $learning_rate $batch_size $model $transform
            done
        done
    done
done