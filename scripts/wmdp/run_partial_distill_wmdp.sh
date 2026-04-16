#!/bin/bash
#
# Run scripts/wmdp/run_partial_distill_wmdp.py via Slurm.
#
# Submit from repo root:
#   sbatch scripts/wmdp/run_partial_distill_wmdp.sh

# --- Slurm Configuration ---
#SBATCH --job-name=wmdp_partial_distill
#SBATCH --output=logs/wmdp_partial_distill_%j.out
#SBATCH --error=logs/wmdp_partial_distill_%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# --- Email Notifications ---
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Environment Setup ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate /home/morg/students/rashkovits/envs/undo

# --- Project Setup ---
cd /home/morg/students/rashkovits/Localized-UNDO
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p logs

# --- Execute Partial Distillation Sweep ---
echo "--------------------------------------------------------"
echo "Starting WMDP Partial Distillation Sweep on Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "--------------------------------------------------------"

python scripts/wmdp/run_partial_distill_wmdp.py

echo "--------------------------------------------------------"
echo "WMDP Partial Distillation Sweep Finished at $(date)"
echo "--------------------------------------------------------"
