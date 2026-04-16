#!/bin/bash

#SBATCH --job-name=download_gemma-2-2b
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=02:00:00               # 2 hours is plenty for download
#SBATCH --partition=studentkillable
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                     # Gemma-2-2b needs ~10-16GB to load safely in RAM

# --- Email Notifications ---
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Environment Setup ---
source ~/.bashrc
conda activate undo

# --- Project Setup ---
# Navigate to the root of the provided repository
cd /home/morg/students/rashkovits/Localized-UNDO
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p logs

# --- Execution ---
echo "--------------------------------------------------------"
echo "Starting Gemma-2-2B Download"
echo "Date: $(date)"
echo "--------------------------------------------------------"

# Crucial: Ensure you have logged in via 'huggingface-cli login' on the login node
# before running this, as custom_login() might require credentials.
python localized_undo/prepare_models/download_gemma.py

echo "--------------------------------------------------------"
echo "Download Finished at $(date)"
echo "--------------------------------------------------------"