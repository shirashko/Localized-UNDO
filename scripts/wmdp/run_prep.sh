#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=gemini_data_prep
#SBATCH --output=logs/prep_%j.out
#SBATCH --error=logs/prep_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=studentkillable
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# --- Email Notifications ---
#SBATCH --mail-user=rashkovits@mail.tau.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Environment Setup ---
source ~/.bashrc
conda activate undo

# --- Project Setup ---
cd /home/morg/students/rashkovits/Localized-UNDO
export PYTHONPATH=$PYTHONPATH:$(pwd)
mkdir -p logs

# Set the path to your Python script
PY_SCRIPT="scripts/wmdp/generate_qa.py"

echo "Starting Data Preparation on Node: $SLURMD_NODENAME"

# --- Stage 1: Generate Forget Set (Bio) ---
# The goal is to create Q&A pairs from "harmful" corpora to train the teacher model on what to forget[cite: 85, 319, 1224].
echo "Running Stage 1: Bio Forget Set Generation"
# Update PROMPT_TYPE to 'bio' in the script automatically
sed -i 's/PROMPT_TYPE = .*/PROMPT_TYPE = "bio"/' $PY_SCRIPT

python3 -c "
from scripts.wmdp.generate_qa import run, concat_jsons, AsyncLimiter
import asyncio

async def main():
    # Use AsyncLimiter to stay within Gemini API rate limits
    limiter = AsyncLimiter(max_rate=3, time_period=1)

    # Generate Forget Set for Biology.
    # Adjust end_idx based on the desired sample size for the Oracle[cite: 96, 1221].
    run(corpus='bio_remove_dataset', batch_size=500, start_idx=0, end_idx=2000,
        limiter=limiter, suitability_threshold=None)

    # Concatenate the generated JSON chunks into a single combined file.
    concat_jsons('wmdp-bio_remove_dataset')

asyncio.run(main())
"

# --- Stage 2: Generate Retain Set (Wikipedia) ---
# The goal is to preserve general knowledge performance (Retain Evaluation)[cite: 87, 624].
echo "Running Stage 2: Wikipedia Retain Set Generation"
# Update PROMPT_TYPE to 'wikipedia'
sed -i 's/PROMPT_TYPE = .*/PROMPT_TYPE = "wikipedia"/' $PY_SCRIPT

python3 -c "
from scripts.wmdp.generate_qa import run, concat_jsons, AsyncLimiter
import asyncio

async def main():
    limiter = AsyncLimiter(max_rate=3, time_period=1)

    # Generate Retain Set from Wikipedia to maintain the model's overall functionality[cite: 80, 1227].
    run(corpus='wikipedia', batch_size=500, start_idx=0, end_idx=2000,
        limiter=limiter, suitability_threshold=None)

    # Concatenate Wikipedia files into a single combined file.
    concat_jsons('wmdp-wikipedia')

asyncio.run(main())
"

echo "Data Preparation Complete. Ready for Unlearn-and-Distill stage."