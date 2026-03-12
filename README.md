# 🔬 Localized UNDO: Targeted Distillation Robustifies Unlearning

## ⚡ Quick Start
For users who want to run a minimal example on language tasks:

1. Complete the [Environment Setup](#-setting-up-environment)
2. Follow [Set up only language](#-set-up-only-language) instructions
3. Run scripts in the following sequence: pretrain, unlearn, partial-distill, relearn

This will get you up and running with the core functionality on a single task type.

## 📋 Prerequisites
- Python 3.8+
- CUDA-compatible GPU(s) recommended. Params set for H200s. For GPU's with less GPU memory, try reducing batch size and increasing gradient accumulation by the same factor.

## 📝 General Notes
- All scripts are meant to be run from scripts directory.
- Most run_* scripts will automatically run on all available GPUs, running several processes in parallel or sequentially as available until all specified settings have been run. To restrict the GPU's, precede the command with `CUDA_VISIBLE_DEVICES={desired devices}`.

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
3. `python localized_undo/prepare_data/download_datasets.py`
4. `python localized-undo/prepare_data/download_arithmetic.py`
5. Generate WMDP question-answer datasets via `wmdp_question_extraction.py`
6. `python localized-undo/prepare_data/prepare.py`

### 🗣️ Set up only language
Run all steps above, but before running step 3, open the file `localized-undo/prepare_data/download_datasets.py` and comment out all calls to `download_dataset` except those that are indicated as required for language setting in the comments (the final two). This will substantially speed up steps 3 and 6. Skip steps 4 and 5.

### 🔢 Set up only arithmetic
Run all steps above, but before running step 3, open the file `localized-undo/prepare_data/download_datasets.py` and comment out all calls to `download_dataset` except those that are indicated as required for arithmetic setting in the comments (the second to last). This will substantially speed up steps 3 and 6. Skip step 5.

### 🧪 Set up only WMDP
Run all steps above, but before running step 3, open the file `localized-undo/prepare_data/download_datasets.py` and comment out all calls to `download_dataset` except those that are indicated as required for WMDP setting in the comments (only the last is not required). This will speed up steps 3 and 6. Skip step 4.

## ▶️ Running Scripts
All scripts can be run using
```
python run_...py
```
