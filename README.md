# 🔬 Localized UNDO: Targeted Distillation Robustifies Unlearning

## ⚡ Quick Start

For users who want to run a minimal example on language tasks:

1. Complete the [Environment Setup](#-setting-up-environment)
2. Follow [Set up only language](#-set-up-only-language) instructions
3. Run scripts in the following sequence: pretrain, unlearn, partial-distill, relearn

This will get you up and running with the core functionality on a single task type.

## 📋 Prerequisites

* Python 3.8+
* CUDA-compatible GPU(s) recommended. Params set for H200s. For GPU's with less GPU memory, try reducing batch size and increasing gradient accumulation by the same factor.

## 📝 General Notes

* All scripts are meant to be run from scripts directory.
* Most run_* scripts will automatically run on all available GPUs, running several processes in parallel or sequentially as available until all specified settings have been run. To restrict the GPU's, precede the command with `CUDA_VISIBLE_DEVICES={desired devices}`.

## 🛠️ Setting Up Environment

1. `git clone https://github.com/shirashko/Localized-UNDO.git`
2. `pip install uv`
3. `cd Localized-UNDO`
4. `uv sync`
5. `source .venv/bin/activate`

## 🚀 Initial Dataset + Model Processing

### ⚙️ Set up for all settings

1. Add a huggingface token to `tokens/hf_token.txt` and a wandb token to `tokens/wandb_token.txt`
2. `python localized_undo/prepare_models/prepare_reduced_gemma.py`
3. `python localized_undo/prepare_data/download_datasets.py --mode all`
4. `python localized_undo/prepare_data/download_arithmetic.py`
5. Generate WMDP question-answer datasets via `wmdp_question_extraction.py`
6. `python localized_undo/prepare_data/prepare.py`

### 🗣️ Set up only language

Run steps 1-2 above, then run step 3 with the language flag:

* `python localized_undo/prepare_data/download_datasets.py --mode language`

This will only download the specific datasets required for language tasks. Finish by running step 6. Skip steps 4 and 5.

### 🔢 Set up only arithmetic

Run steps 1-2 above, then run step 3 with the arithmetic flag:

* `python localized_undo/prepare_data/download_datasets.py --mode arithmetic`

Followed by step 4 and 6. Skip step 5.

### 🧪 Set up only WMDP

Run steps 1-2 above, then run step 3 with the WMDP flag:

* `python localized_undo/prepare_data/download_datasets.py --mode wmdp`

Followed by step 5 and 6. Skip step 4.

## ▶️ Running Scripts

All scripts can be run using

```bash
python run_...py

```
