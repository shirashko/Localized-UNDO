#!/bin/bash
# Tokenize raw / QA JSONLs into datasets/pretrain/ via prepare.py.
#
# Slurm:  sbatch scripts/wmdp/sbatch_prepare.sh
# Local:  bash scripts/wmdp/run_prepare.sh   (from repo root)
#
# Requires: tokens/hf_token.txt

set -euo pipefail

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
conda activate undo

REPO_ROOT="/home/morg/students/rashkovits/Localized-UNDO"
cd "$REPO_ROOT"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
mkdir -p logs

echo "--------------------------------------------------------"
echo "prepare.py — build datasets/pretrain/"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Started: $(date)"
echo "--------------------------------------------------------"

python localized_undo/prepare_data/prepare.py

echo "--------------------------------------------------------"
echo "Finished: $(date)"
echo "--------------------------------------------------------"
