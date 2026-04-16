#!/bin/bash
#
# Run scripts/wmdp/run_eval.py via Slurm.
#
# Submit from repo root:
#   sbatch scripts/wmdp/run_eval.sh

# --- Slurm Configuration ---
#SBATCH --job-name=wmdp_eval
#SBATCH --output=logs/wmdp_eval_%j.out
#SBATCH --error=logs/wmdp_eval_%j.err
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

MODEL_PATH="/home/morg/students/rashkovits/Localized-UNDO/models/wmdp/gemma-2-2b"

# --- Execute Evaluation ---
echo "--------------------------------------------------------"
echo "Starting WMDP/MMLU Evaluation on Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "Model: $MODEL_PATH"
echo "--------------------------------------------------------"

python scripts/wmdp/run_eval.py \
  --model-name "$MODEL_PATH" \
  --domain bio \
  --large-eval \
  --full-mmlu \
  --report-all-subtasks \
  --report-mmlu-bio-breakdown

echo "--------------------------------------------------------"
echo "WMDP/MMLU Evaluation Finished at $(date)"
echo "--------------------------------------------------------"
