#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=distill_sweep
#SBATCH --output=logs/distill_%j.out
#SBATCH --error=logs/distill_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# --- Email Notifications ---
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Parameters ---
# Change this variable to run different setups from your YAML
SETUP_ID="gemma-2-0.3B_MaxEnt"
CONFIG_FILE="configs/arithmetic/partial_distill.yaml"

# --- Environment Setup ---
source ~/.bashrc
conda activate undo

# --- Project Setup ---
cd /home/morg/students/rashkovits/Localized-UNDO
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p logs

# --- Execute Partial Distillation Sweep ---
echo "--------------------------------------------------------"
echo "Starting Partial Distillation Sweep on Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Target Setup ID: $SETUP_ID"
echo "--------------------------------------------------------"

# Print the full configuration for the record
echo "[*] Full Experiment Configuration ($CONFIG_FILE):"
cat "$CONFIG_FILE"
echo "--------------------------------------------------------"

python scripts/arithmetic/run_partial_distill.py --setup "$SETUP_ID"

echo "--------------------------------------------------------"
echo "Partial Distillation Sweep ($SETUP_ID) Finished at $(date)"
echo "--------------------------------------------------------"