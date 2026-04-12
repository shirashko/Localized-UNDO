#!/bin/bash
# Run scripts/wmdp/wmdp_question_extraction.py (Gemini QA extraction / concat_jsons).
#
# Requires:
#   - tokens/gemini_token.txt
#   - Input JSONLs under datasets/wmdp/ when using the Gemini `run()` path (see script __main__)
#
# Submit from repo root:
#   sbatch scripts/wmdp/run_wmdp_question_extraction.sh

#SBATCH --job-name=wmdp_qa_extract
#SBATCH --output=logs/wmdp_question_extraction_%j.out
#SBATCH --error=logs/wmdp_question_extraction_%j.err
#SBATCH --partition=studentkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

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

QA_SCRIPT="${QA_SCRIPT:-scripts/wmdp/wmdp_question_extraction.py}"

echo "--------------------------------------------------------"
echo "WMDP question extraction"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Script: $QA_SCRIPT"
echo "Started: $(date)"
echo "--------------------------------------------------------"

python "$QA_SCRIPT"

echo "--------------------------------------------------------"
echo "Finished: $(date)"
echo "--------------------------------------------------------"
