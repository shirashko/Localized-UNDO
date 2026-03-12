#!/bin/bash
#SBATCH --job-name=download_wmdp
#SBATCH --output=logs/download_wmdp_%j.out
#SBATCH --error=logs/download_wmdp_%j.err
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# 1. Activate the environment
source activate undo

# 2. Ensure logs directory exists to avoid write errors
mkdir -p logs

# 3. Execute the download script
# Using --mode wmdp will download:
# - All Magpie alignment datasets (Llama, Qwen, Phi, Gemma)
# - Wikitext and Wikipedia
# - The shared English FineWeb-EDU sample
python download_datasets.py --mode wmdp

# 4. Success message in the log file
echo "Download process for WMDP completed at $(date)"