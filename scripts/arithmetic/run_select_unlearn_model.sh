#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=select_unlearn_model
#SBATCH --output=logs/select_unlearn_%j.out
#SBATCH --error=logs/select_unlearn_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# --- Email Notifications ---
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Environment Setup ---
source ~/.bashrc
conda activate undo

# --- Project Setup ---
cd /home/morg/students/rashkovits/Localized-UNDO
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p logs

echo "--------------------------------------------------------"
echo "Starting unlearn sweep selection on node: $SLURMD_NODENAME"
echo "Script target: scripts/arithmetic/select_unlearn_model.py"
echo "--------------------------------------------------------"

echo "[*] Active unlearning config:"
cat configs/arithmetic/unlearn.yaml
echo "--------------------------------------------------------"

python scripts/arithmetic/select_unlearn_model.py

echo "--------------------------------------------------------"
echo "Selection run finished at $(date)"
