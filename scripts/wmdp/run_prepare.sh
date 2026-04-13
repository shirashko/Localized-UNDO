#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=prepare_data
#SBATCH --output=logs/prepare_data_%j.out
#SBATCH --error=logs/prepare_data_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# --- Email Notifications ---
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# --- Environment Setup ---
# Align with other Slurm scripts so cluster-level conda init is loaded.
if [ -f "$HOME/.bashrc" ]; then
  # shellcheck source=/dev/null
  source "$HOME/.bashrc"
fi

_conda_base="${CONDA_ROOT:-}"
if [ -z "$_conda_base" ] || [ ! -f "$_conda_base/etc/profile.d/conda.sh" ]; then
  _conda_base=""
  for d in \
    "$HOME/miniconda3" "$HOME/mambaforge" "$HOME/miniforge3" "$HOME/anaconda3" \
    "/home/morg/students/$USER/miniconda3" "/home/morg/students/$USER/mambaforge" \
    "/home/morg/students/$USER/miniforge3" "/home/morg/students/$USER/anaconda3"; do
    if [ -f "$d/etc/profile.d/conda.sh" ]; then
      _conda_base="$d"
      break
    fi
  done
fi
if [ -z "$_conda_base" ]; then
  echo "ERROR: conda.sh not found. Set CONDA_ROOT." >&2
  exit 1
fi
# shellcheck source=/dev/null
. "$_conda_base/etc/profile.d/conda.sh"
unset _conda_base
CONDA_ENV_NAME="${CONDA_ENV_NAME:-undo}"
conda activate "$CONDA_ENV_NAME"

# --- Project Setup ---
REPO_ROOT="/home/morg/students/rashkovits/Localized-UNDO"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
mkdir -p logs

# --- Execute Prepare ---
echo "--------------------------------------------------------"
echo "prepare.py — build datasets/pretrain/"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Started: $(date)"
echo "--------------------------------------------------------"

python localized_undo/prepare_data/prepare.py

echo "--------------------------------------------------------"
echo "Finished: $(date)"
echo "--------------------------------------------------------"
