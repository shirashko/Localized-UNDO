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

set -euo pipefail

# Conda (avoid relying on cwd; compute nodes may differ from $HOME layout)
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

# Repo root (submit from anywhere)
cd /home/morg/students/rashkovits/Localized-UNDO
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
mkdir -p logs
python localized_undo/prepare_data/download_datasets.py --mode wmdp

echo "Download process for WMDP completed at $(date)"
