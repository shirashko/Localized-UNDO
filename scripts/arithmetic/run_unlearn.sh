#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=unlearn_sweep
#SBATCH --output=logs/unlearn_%j.out
#SBATCH --error=logs/unlearn_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

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

# --- Execute Unlearning Sweep ---
# --- Execute Unlearning Sweep ---
echo "--------------------------------------------------------"
echo "Starting Unlearning Sweep on Node: $SLURMD_NODENAME"
echo "Target Setups: $BASE_SETUPS_TO_RUN"
echo "--------------------------------------------------------"

echo "[*] Full Experiment Configuration (unlearn.yaml):"
cat configs/arithmetic/unlearn.yaml
echo "--------------------------------------------------------"

python scripts/arithmetic/run_unlearn.py

echo "--------------------------------------------------------"
echo "Unlearning Sweep Finished at $(date)"