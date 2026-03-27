import gc
import json
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict
from transformers import AutoModelForCausalLM
from localized_undo.utils.paths import MODEL_DIR, LOCALIZATION_MASKS_DIR, DATASET_DIR
from localized_undo.masks_utils.directional_mask_factory import DirectionalMaskFactory
from localized_undo.masks_utils.weight_surgeon import WeightSurgeon


def generate_variance_plot(all_k_stats: Dict[int, Dict[str, float]]):
    """Generates a plot showing Explained Variance per layer for each k value."""
    plt.figure(figsize=(10, 6))
    for k, layer_stats in all_k_stats.items():
        layer_indices = []
        values = []
        for name, var in layer_stats.items():
            try:
                idx = int(name.split('.')[1])
                layer_indices.append(idx)
                values.append(var)
            except (IndexError, ValueError):
                continue
        sorted_pairs = sorted(zip(layer_indices, values))
        x, y = zip(*sorted_pairs)
        plt.plot(x, y, marker='o', linestyle='-', label=f'k={k}')

    plt.xlabel('Layer Index')
    plt.ylabel('Explained Variance (SVD)')
    plt.title('Unlearning Localization: Explained Variance per Layer')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(LOCALIZATION_MASKS_DIR / "layer_variance_localization.png")
    plt.close()


def generate_tradeoff_plot(sweep_results: List[Dict]):
    """Generates a tradeoff plot: Accuracy and Language Loss vs k."""
    # Sort results by k
    sweep_results = sorted(sweep_results, key=lambda x: x['k'])
    ks = [r['k'] for r in sweep_results]

    # Extract metrics - adjust keys if your Diagnostic uses different names
    acc = [r['metrics'].get('val/multiplication_equation_acc', 0) for r in sweep_results]
    loss = [r['metrics'].get('val/eng_ce_loss', 0) for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('k (Singular Vectors)')
    ax1.set_ylabel('Multiplication Accuracy', color=color)
    ax1.plot(ks, acc, 'o-', color=color, linewidth=2, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('English CE Loss', color=color)
    ax2.plot(ks, loss, 's--', color=color, linewidth=2, label='Language Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Trade-off: Knowledge Erasure vs. Language Stability')
    fig.tight_layout()
    plt.savefig(LOCALIZATION_MASKS_DIR / "accuracy_loss_tradeoff.png")
    plt.close()


def run_directional_ablation_sweep(
        pretrained_path: str,
        unlearned_path: str,
        k_range: List[int],
        exclusions: List[str],
        eval_eng_path: str
):
    print(f"[*] Starting Comprehensive SVD Sweep. k_range: {k_range}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results_for_plot = []

    # --- Step 1: Load models & Diagnostics ---
    ref_model = AutoModelForCausalLM.from_pretrained(pretrained_path, dtype=torch.float32).to("cpu")
    model = AutoModelForCausalLM.from_pretrained(unlearned_path, dtype=torch.float32).to(device)
    surgeon = WeightSurgeon(model)
    from localized_undo.utils.localization_diagnoser import MaskMechanisticDiagnostic
    analyzer = MaskMechanisticDiagnostic(unlearned_path, eval_eng_path, device=device)

    # --- Step 2: Baseline (No Masking) ---
    print("\n>>> [BASELINE] Evaluating Unlearned model weights AS-IS...")
    with torch.inference_mode():
        baseline_metrics = analyzer.eval_fn(model, print_results=False)

    baseline_folder = LOCALIZATION_MASKS_DIR / "baseline_no_mask"
    baseline_folder.mkdir(parents=True, exist_ok=True)
    with open(baseline_folder / "experiment_info.json", "w") as f:
        json.dump({"k": 0, "description": "Raw unlearned model", "metrics": baseline_metrics}, f, indent=4)

    # --- Step 3: Pre-compute SVD masks ---
    ref_params = {n.replace("model.", ""): p for n, p in ref_model.named_parameters()}
    all_k_masks, all_k_stats = {}, {}
    for k in k_range:
        current_k_masks, current_k_stats = {}, {}
        for name, param in model.named_parameters():
            clean_name = name.replace("model.", "")
            if DirectionalMaskFactory.is_target_layer(clean_name, exclusions):
                p_mat, var = DirectionalMaskFactory.compute_projection_mask(param.data, ref_params[clean_name], k)
                current_k_masks[clean_name] = p_mat.cpu()
                current_k_stats[clean_name] = var
        all_k_masks[k] = current_k_masks
        all_k_stats[k] = current_k_stats

    del ref_model
    gc.collect();
    torch.cuda.empty_cache()

    # --- Step 4: Random Masking Control (k=1) ---
    print("\n>>> [CONTROL] Evaluating Random Masking (k=1 representative)...")
    random_k = 1
    random_masks = {}
    for name, param in model.named_parameters():
        clean_name = name.replace("model.", "")
        if DirectionalMaskFactory.is_target_layer(clean_name, exclusions):
            random_masks[clean_name] = DirectionalMaskFactory.compute_random_mask(param.data, random_k).cpu()

    surgeon.apply_masks(random_masks)
    with torch.inference_mode():
        random_metrics = analyzer.eval_fn(model, print_results=False)

    random_folder = LOCALIZATION_MASKS_DIR / "control_random_k1"
    random_folder.mkdir(parents=True, exist_ok=True)
    with open(random_folder / "experiment_info.json", "w") as f:
        json.dump({"k": random_k, "description": "Random directions projected out", "metrics": random_metrics}, f,
                  indent=4)

    surgeon.restore()  # מחזירים את המודל למצב Unlearned נקי לפני ה-Sweep
    torch.cuda.empty_cache()

    # --- Step 5: Main Sweep Loop ---
    for k in k_range:
        print(f"\n>>> Evaluating Directional Ablation (k={k})")
        surgeon.apply_masks(all_k_masks[k])
        with torch.inference_mode():
            metrics = analyzer.eval_fn(model, print_results=False)

        res = {"k": k, "metrics": metrics, "avg_variance_explained": np.mean(list(all_k_stats[k].values()))}
        all_results_for_plot.append(res)

        k_folder = LOCALIZATION_MASKS_DIR / f"svd_sweep_k{k}"
        k_folder.mkdir(parents=True, exist_ok=True)
        with open(k_folder / "experiment_info.json", "w") as f:
            json.dump(res, f, indent=4)

        surgeon.restore()
        torch.cuda.empty_cache()

    # --- Step 6: Visualization ---
    print("[*] Generating summary plots...")
    generate_variance_plot(all_k_stats)
    generate_tradeoff_plot(all_results_for_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs='+', default=[1, 2, 5, 10], help="List of k values")
    args = parser.parse_args()

    # Verified absolute paths
    PRETRAINED_PATH = str(MODEL_DIR / "pretrained_models" / "gemma-2-0.3B_all_arithmetic+eng" / "final_model")

    UNLEARNED_PATH = "/home/morg/students/rashkovits/Localized-UNDO/models/non-wmdp/unlearned_models/MAXEnt_prev_02/pretrained_models_gemma-2-0.3B_all_arithmetic+eng_final_model_lr_1.0e-04/final_model"

    run_directional_ablation_sweep(
        pretrained_path=PRETRAINED_PATH,
        unlearned_path=UNLEARNED_PATH,
        k_range=args.k,
        exclusions=["self_attn", "layernorm", "embed_tokens"],
        eval_eng_path=str(DATASET_DIR / "pretrain" / "valid_eng.jsonl")
    )