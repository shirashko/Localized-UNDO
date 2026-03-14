import os
import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM

from localized_undo.utils.paths import MODEL_DIR, LOCALIZATION_MASKS_DIR
from localized_undo.utils.mask_factory import MaskFactory


def create_mask(
        reference_model_path: str,
        unlearned_model_path: str,
        percentile: float = 0.1,
        exclude_components: list = ["self_attn", "layernorm", "embed_tokens"],
        base_folder_name: str = "localization"
):
    """
    Creates a Delta mask and a matching Random mask baseline under the same constraints.
    Saves both .pt files and a shared metadata .json file.
    """
    print(f"[*] Initializing Paired Mask Creation (Target Density: {percentile})...")
    device = torch.device("cpu")

    # 1. Load Models
    ref_model = AutoModelForCausalLM.from_pretrained(reference_model_path, torch_dtype=torch.float32, device_map="cpu")
    unlearned_model = AutoModelForCausalLM.from_pretrained(unlearned_model_path, torch_dtype=torch.float32,
                                                           device_map="cpu")

    # 2. Generate Masks
    mask_results = {}
    for mask_type in ["delta_mask", "random"]:
        print(f"[*] Generating {mask_type}...")
        mask = MaskFactory.get_mask(
            mask_type=mask_type,
            model=unlearned_model,
            ref_model=ref_model if mask_type == "delta_mask" else None,
            percentile=percentile,
            exclude_components=exclude_components,
            device=device
        )
        mask_results[mask_type] = mask

    # 3. Validation & Metadata Preparation
    delta_stats = MaskFactory.analyze_mask(mask_results["delta_mask"], unlearned_model)
    random_stats = MaskFactory.analyze_mask(mask_results["random"], unlearned_model)

    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "percentile": percentile,
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

    # 4. Save Outputs
    folder_name = f"{base_folder_name}_p{int(percentile * 100)}_excl_{'_'.join(exclude_components) if exclude_components else 'none'}"
    output_dir = LOCALIZATION_MASKS_DIR / folder_name
    os.makedirs(output_dir, exist_ok=True)

    # Save Delta Mask
    torch.save(mask_results["delta_mask"], output_dir / "delta_mask.pt")
    # Save Random Mask baseline
    torch.save(mask_results["random"], output_dir / "random_baseline.pt")
    # Save Metadata
    with open(output_dir / "mask_config.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[+] Paired masks saved to: {output_dir}")
    print(f"    - Delta Sparsity: {metadata['masks']['delta']['sparsity']}")
    print(f"    - Random Sparsity: {metadata['masks']['random']['sparsity']}")

    return output_dir


if __name__ == "__main__":
    # Configure Paths
    PRETRAINED_PATH = str(MODEL_DIR / "pretrained_models" / "gemma-2-0.1B_all_arithmetic+eng" / "final_model")
    UNLEARNED_PATH = str(
        MODEL_DIR / "unlearned_models" / "MaxEnt" / "pretrained_models_gemma-2-0.1B_all_arithmetic+eng_final_model_lr_8.0e-05" / "final_model")

    # Hyperparameters for this specific run
    percentile = 0.2
    EXCLUSIONS = ["self_attn", "layernorm", "embed_tokens"]

    create_mask(
        reference_model_path=PRETRAINED_PATH,
        unlearned_model_path=UNLEARNED_PATH,
        percentile=percentile,
        exclude_components=EXCLUSIONS,
        base_folder_name="arithmetic"
    )