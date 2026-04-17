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
# Request NVIDIA (e.g. H100). Plain `--gres=gpu:1` can land on AMD nodes (e.g. n-210);
# PyTorch `torch.cuda` then sees 0 GPUs — use a CUDA-capable GRES name from `sinfo -p gpu-morgeva`.
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=24:00:00

# --- Email Notifications ---
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Environment Setup ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate /home/morg/students/rashkovits/envs/undo

# Reduce allocator fragmentation after large evals (lm-eval) + training peaks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Project Setup ---
cd /home/morg/students/rashkovits/Localized-UNDO
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p logs

# --- GPU diagnostics (see this job's .out if "No GPU devices found") ---
echo "--------------------------------------------------------"
echo "GPU diagnostics (job $SLURM_JOB_ID on $SLURMD_NODENAME)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  nvidia-smi -L || true
else
  echo "nvidia-smi: not found in PATH"
fi
python -c "import torch; print('torch.cuda.is_available:', torch.cuda.is_available()); print('torch.cuda.device_count:', torch.cuda.device_count()); print('torch.version.cuda:', torch.version.cuda)" || true
echo "--------------------------------------------------------"

# --- Execute Partial Distillation Sweep ---
echo "--------------------------------------------------------"
echo "Starting WMDP Partial Distillation Sweep on Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "--------------------------------------------------------"

python scripts/wmdp/run_partial_distill_wmdp.py

echo "--------------------------------------------------------"
echo "WMDP Partial Distillation Sweep Finished at $(date)"
echo "--------------------------------------------------------"
