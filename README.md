## 🔬 Localized UNDO: Targeted Noise Eases Distillation Compute while Remaining Robust Unlearning

This repository contains the code and instructions to reproduce the experiments from our paper on localized unlearning via targeted noise injection. The project is structured around two main experimental settings: **WMDP (Bio/Cyber)** and **Arithmetic**.

---

### 📋 Prerequisites
* **Python:** 3.8+.
* **Hardware:** CUDA-compatible GPU(s) recommended. Params are optimized for **H100/H200 GPUs**. For consumer GPUs (e.g., RTX 2080/3090/4090), you **must** reduce `batch_size` and increase `gradient_accumulation_steps` proportionally to avoid Out-Of-Memory (OOM) errors.

## 🛠️ Setting Up Environment

1. `git clone https://github.com/shirashko/Localized-UNDO.git`
2. `pip install uv`
3. `cd Localized-UNDO`
4. `uv sync`
5. `source .venv/bin/activate`

### 💡 General Tips
* **Authentication:** Add your Hugging Face token to `tokens/hf_token.txt` and WandB token to `tokens/wandb_token.txt`.
* **Gated Models:** [Gemma-2-2b](https://huggingface.co/google/gemma-2-2b) requires manual license acceptance on Hugging Face before download scripts will work.
* **Parallel Launch:** Most scripts use a `launch_in_parallel_one_per_gpu` utility to automatically run multiple hyperparameter experiments (e.g., different learning rates) simultaneously across available GPUs.
* **Monitoring:** Integrated with **Weights & Biases (WandB)** to log "Forget" and "Retain" set metrics in real-time.
---

### 🧪 Setting: WMDP (Bio/Cyber Unlearning)
Use this to reproduce unlearning of hazardous knowledge from the Weapon-Masked Data Poisoning (WMDP) benchmark.

#### 1. Setup & Data Prep
1.  **Download Model:** `python localized_undo/prepare_models/download_gemma.py`.
2.  **Download Base Data:** `python localized_undo/prepare_data/download_datasets.py --mode wmdp`.
3.  **Generate QA Pairs:** `python wmdp_question_extraction.py` (Requires `GOOGLE_API_KEY` for Gemini to generate the synthetic forget/retain corpora).
4.  **Finalize:** `python localized_undo/prepare_data/prepare.py`.

#### 2. Execution Sequence
Run these from the **root directory**:
1.  **Unlearn:** `python run_unlearn_wmdp.py` — Hyperparameter sweep for methods like **MaxEnt** and **RMU**.
2.  **Select:** `python select_unlearn_model.py` — Identifies the optimal checkpoint balancing forget-set removal with retain-set performance.
3.  **Distill (UNDO):** `python run_partial_distill_wmdp.py` — The core step distilling the unlearned model into a noised copy to scrub latent capabilities.
4.  **Relearn:** `python run_relearn_wmdp.py` — Evaluates robustness by attempting to recover forgotten knowledge via adversarial fine-tuning.

---

### 🔢 Setting: Arithmetic
Forget set = multiplication and division (equations and word problems).
Retain set = addition and subtraction (equations and word problems).

#### 1. Setup & Data Prep
1.  **Prepare Model:** `python localized_undo/prepare_models/reduce_gemma.py` (Creates a smaller version of Gemma for faster iteration).
2.  **Download Data:** `python localized_undo/prepare_data/download_datasets.py --mode arithmetic`.
3.  **Generate Math:** `python localized_undo/prepare_data/download_arithmetic.py`.
4.  **Finalize:** `python localized_undo/prepare_data/prepare.py`.

#### 2. Execution Sequence
1.  **Pretrain:** `python run_pretrain_arithmetic.py` — Establishes the initial model with math capabilities.
2.  **Unlearn:** `python run_unlearn_arithmetic.py`.
3.  **Distill:** `python run_partial_distill_arithmetic.py`.
4.  **Relearn:** `python run_relearn_arithmetic.py`.

