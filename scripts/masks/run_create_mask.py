import os
import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM
from localized_undo.utils.paths import MODEL_DIR, LOCALIZATION_MASKS_DIR
from localized_undo.utils.mask_factory import MaskFactory


def create_mask_sweep(
        reference_model_path: str,
        unlearned_model_path: str,
        percentiles: list[float],
        exclude_components: list = ["self_attn", "layernorm", "embed_tokens"],
        base_folder_name: str = "arithmetic"
):
    """
    Loads models once and generates paired Delta and Random masks for a sweep of percentiles.
    """
    print(f"[*] Starting Mask Sweep for percentiles: {percentiles}")
    device = torch.device("cpu")

    # 1. Load Models Once (expensive step)
    print(f"[*] Loading models to CPU...")
    ref_model = AutoModelForCausalLM.from_pretrained(reference_model_path, torch_dtype=torch.float32, device_map="cpu")
    unlearned_model = AutoModelForCausalLM.from_pretrained(unlearned_model_path, torch_dtype=torch.float32,
                                                           device_map="cpu")

    # 2. Iterate through each percentile in the sweep
    for p in percentiles:
        print(f"\n--- Generating Masks for Percentile: {p} ---")

        mask_results = {}
        for mask_type in ["delta_mask", "random"]:
            mask = MaskFactory.get_mask(
                mask_type=mask_type,
                model=unlearned_model,
                ref_model=ref_model if mask_type == "delta_mask" else None,
                percentile=p,
                exclude_components=exclude_components,
                device=device
            )
            mask_results[mask_type] = mask

        # 3. Analyze & Metadata
        delta_stats = MaskFactory.analyze_mask(mask_results["delta_mask"], unlearned_model)
        random_stats = MaskFactory.analyze_mask(mask_results["random"], unlearned_model)

        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "percentile": p,
                "exclude_components": exclude_components,
                "reference_model": reference_model_path,
                "unlearned_model": unlearned_model_path,
            },
            "masks": {
                "delta": {
                    "sparsity": f"{delta_stats['mask/total_sparsity']:.4%}",
                    "weights_count": delta_stats['mask/masked_count']
                },
                "random": {
                    "sparsity": f"{random_stats['mask/total_sparsity']:.4%}",
                    "weights_count": random_stats['mask/masked_count']
                }
            }
        }

        # 4. Save Outputs in distinct folders
        p_int = int(p * 100) if p < 1.0 else int(p)
        folder_name = f"{base_folder_name}_p{p_int}_excl_{'_'.join(exclude_components) if exclude_components else 'none'}"
        output_dir = LOCALIZATION_MASKS_DIR / folder_name
        os.makedirs(output_dir, exist_ok=True)

        torch.save(mask_results["delta_mask"], output_dir / "delta_mask.pt")
        torch.save(mask_results["random"], output_dir / "random_baseline.pt")

        with open(output_dir / "mask_config.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[+] Saved percentile {p} to: {output_dir}")

    print(f"\n[SUCCESS] Completed sweep of {len(percentiles)} masks.")


if __name__ == "__main__":
    # Configure Paths
    PRETRAINED_PATH = str(MODEL_DIR / "pretrained_models" / "gemma-2-0.1B_all_arithmetic+eng" / "final_model")
    UNLEARNED_PATH = str(
        MODEL_DIR / "unlearned_models" / "MaxEnt" / "pretrained_models_gemma-2-0.1B_all_arithmetic+eng_final_model_lr_8.0e-05" / "final_model")

    # Define Sweep Hyperparameters
    PERCENTILE_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.5]  # Sweep from 5% to 50% density
    EXCLUSIONS = ["self_attn", "layernorm", "embed_tokens"]

    create_mask_sweep(
        reference_model_path=PRETRAINED_PATH,
        unlearned_model_path=UNLEARNED_PATH,
        percentiles=PERCENTILE_SWEEP,
        exclude_components=EXCLUSIONS,
        base_folder_name="arithmetic"
    )