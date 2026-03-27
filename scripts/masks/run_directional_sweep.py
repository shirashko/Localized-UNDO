import os
import gc
import json
import torch
import numpy as np
import argparse
from typing import List
from transformers import AutoModelForCausalLM
from localized_undo.utils.paths import MODEL_DIR, LOCALIZATION_MASKS_DIR, DATASET_DIR
from localized_undo.masks_utils.directional_mask_factory import DirectionalMaskFactory
from localized_undo.masks_utils.weight_surgeon import WeightSurgeon


def run_directional_ablation_sweep(
        pretrained_path: str,
        unlearned_path: str,
        k_range: List[int],
        exclusions: List[str],
        eval_eng_path: str
):
    print(f"[*] Starting SVD Sweep on k: {k_range}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Verify paths
    pretrained_path = os.path.abspath(pretrained_path)
    unlearned_path = os.path.abspath(unlearned_path)
    for p in [pretrained_path, unlearned_path]:
        if not os.path.exists(p):
            raise OSError(f"Directory not found: {p}")

    # 1. Load models - Reference stays on CPU to save VRAM
    print("[*] Loading reference model to CPU...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        pretrained_path, dtype=torch.float32, local_files_only=True
    ).to("cpu")

    print(f"[*] Loading unlearned model to {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        unlearned_path, dtype=torch.float32, local_files_only=True
    ).to(device)

    # 2. Pre-compute ALL masks in CPU RAM
    print("[*] Pre-computing SVD masks for all k values...")
    all_k_masks = {}
    all_k_stats = {}
    ref_params = {n.replace("model.", ""): p for n, p in ref_model.named_parameters()}

    for k in k_range:
        current_k_masks = {}
        current_k_stats = {}
        for name, param in model.named_parameters():
            clean_name = name.replace("model.", "")
            if DirectionalMaskFactory.is_target_layer(clean_name, exclusions):
                # We do SVD on CPU/GPU depending on where param is, but move result to CPU
                p_mat, var = DirectionalMaskFactory.compute_projection_mask(
                    param.data, ref_params[clean_name], k
                )
                current_k_masks[clean_name] = p_mat.cpu()
                current_k_stats[clean_name] = var
        all_k_masks[k] = current_k_masks
        all_k_stats[k] = current_k_stats

    # 3. Aggressive Memory Cleanup
    del ref_model
    del ref_params
    gc.collect()
    torch.cuda.empty_cache()
    print("[*] Reference model purged. GPU memory cleared for evaluation.")

    # 4. Setup Surgeon and Diagnostic
    surgeon = WeightSurgeon(model)
    from localized_undo.utils.localization_diagnoser import MaskMechanisticDiagnostic
    analyzer = MaskMechanisticDiagnostic(unlearned_path, eval_eng_path, device=device)

    # 5. Execution Loop
    for k in k_range:
        print(f"\n>>> Evaluating Directional Ablation (k={k})")

        # Apply masks from CPU to GPU Just-In-Time
        surgeon.apply_masks(all_k_masks[k])

        # Diagnostic Evaluation in Inference Mode
        with torch.inference_mode():
            # This is where the OOM happened - now we have more headroom
            metrics = analyzer.eval_fn(model, print_results=False)

        # Save Results
        k_folder = LOCALIZATION_MASKS_DIR / f"svd_sweep_k{k}"
        k_folder.mkdir(parents=True, exist_ok=True)

        with open(k_folder / "experiment_info.json", "w") as f:
            json.dump({
                "k": k,
                "avg_variance_explained": np.mean(list(all_k_stats[k].values())),
                "metrics": metrics,
                "layer_details": all_k_stats[k]
            }, f, indent=4)

        torch.save(all_k_masks[k], k_folder / "mask.pt")

        # Cleanup for next iteration
        surgeon.restore()
        torch.cuda.empty_cache()
        print(f"[SUCCESS] Results for k={k} saved.")


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