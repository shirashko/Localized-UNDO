#!/bin/bash
#SBATCH --job-name=download_wmdp_bio_corpus
#SBATCH --output=logs/download_wmdp_bio_corpus_%j.out
#SBATCH --error=logs/download_wmdp_bio_corpus_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G

# --- Email Notifications ---
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Environment Setup ---
source ~/.bashrc
conda activate undo

# --- Project root ---
cd /home/morg/students/rashkovits/Localized-UNDO
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p logs

echo "--------------------------------------------------------"
echo "WMDP bio forget/retain corpus download"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Started: $(date)"
echo "--------------------------------------------------------"

python localized_undo/prepare_data/download_wmdp_bio_corpora.py ${EXTRA_ARGS:-}

echo "--------------------------------------------------------"
echo "Finished: $(date)"
echo "Next (on login or another job): python localized_undo/prepare_data/prepare.py"
echo "--------------------------------------------------------"
