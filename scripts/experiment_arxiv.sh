#!/bin/bash

## Run standing/not standing experiment next
## Run other learning rates after
learning_rates=(0.00025)
batch_sizes=(64)
model=("vgg16")
transforms=("RandomErasing_RandomHorizontalFlip_RandomVerticalFlip")
category=("pose")

# create an array with all variables above


# Double nested for loop
## loop 30 times
for i in {1..30}; do
    for model in "${model[@]}"; do
        for category in "${category[@]}"; do
            for learning_rate in "${learning_rates[@]}"; do
                for batch_size in "${batch_sizes[@]}"; do
                    for transform in "${transforms[@]}"; do
                        sbatch -J "lr$learning_rate-bs$batch_size-model$model-transforms$transform-category$category" scripts/runmodeltemplate.sh $learning_rate $batch_size $model $transform $category $i
                    done
                done
            done
        done
    done
done