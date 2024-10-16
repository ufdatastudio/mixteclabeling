#!/bin/bash
#SBATCH --job-name=name_date_classify_%j
#SBATCH --output=logs/name_date_classify_%j.out
#SBATCH --error=logs/name_date_classify_%j.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=gsalunke@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000mb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00

mkdir -p logs

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# Load mamba and activate environment
module load mamba
mamba activate name-date || { echo "Failed to activate environment"; exit 1; }

T1=$(date +%s)

# Run the Python script
python classification.py

T2=$(date +%s)

mamba deactivate

ELAPSED=$((T2 - T1))
echo "Elapsed time: $(($ELAPSED / 3600))h $((($ELAPSED / 60) % 60))m $(($ELAPSED % 60))s"
