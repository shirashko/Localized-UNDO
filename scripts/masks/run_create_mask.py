import os
import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM

from localized_undo.utils.paths import MODEL_DIR, LOCALIZATION_MASKS_DIR
from localized_undo.utils.mask_factory import MaskFactory


def create_and_save_localization_mask(
        reference_model_path: str,
        unlearned_model_path: str,
        output_path: str,
        percentile: float = 0.1,
        exclude_components: list = ["self_attn", "layernorm", "embed_tokens"]
):
    """
    Loads two models, computes the weight-level delta mask, and saves it to disk
    alongside a JSON config file containing the generation metadata.
    """
    print(f"[*] Initializing creation of mask (Percentile: {percentile})...")

    # Use CPU for mask generation to avoid VRAM overhead
    device = torch.device("cpu")

    # 1. Load Models
    print(f"[*] Loading reference model from: {reference_model_path}")
    ref_model = AutoModelForCausalLM.from_pretrained(reference_model_path, torch_dtype=torch.float32, device_map="cpu")

    print(f"[*] Loading unlearned model from: {unlearned_model_path}")
    unlearned_model = AutoModelForCausalLM.from_pretrained(unlearned_model_path, torch_dtype=torch.float32,
                                                           device_map="cpu")

    # 2. Verify naming consistency
    MaskFactory.debug_naming_mismatch(unlearned_model, ref_model)

    # 3. Generate Delta Mask
    print(f"[*] Generating Delta Mask...")
    delta_mask = MaskFactory.get_mask(
        mask_type="delta_mask",
        model=unlearned_model,
        ref_model=ref_model,
        percentile=percentile,
        exclude_components=exclude_components,
        device=device
    )

    # 4. Analyze results for metadata
    stats = MaskFactory.analyze_mask(delta_mask, unlearned_model)

    # 5. Prepare and Save Metadata Config
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "percentile": percentile,
        "exclude_components": exclude_components,
        "reference_model": reference_model_path,
        "unlearned_model": unlearned_model_path,
        "mask_statistics": {
            "total_sparsity": f"{stats['mask/total_sparsity']:.4%}",
            "active_layers": stats['mask/active_layers'],
            "masked_weights_count": stats.get('mask/masked_count')
        }
    }

    # Define paths
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    config_save_path = str(output_path).replace(".pt", ".json")

    # 6. Save the Resulting Mask and Config
    print(f"[*] Saving Delta Mask to {output_path}...")
    torch.save(delta_mask, output_path)

    print(f"[*] Saving Mask Metadata to {config_save_path}...")
    with open(config_save_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[+] Mask and Config generated successfully.")
    print(f"    - Total Sparsity: {metadata['mask_statistics']['total_sparsity']}")

    return output_path


if __name__ == "__main__":
    PRETRAINED_PATH = str(MODEL_DIR / "pretrained_models" / "gemma-2-0.1B_all_arithmetic+eng" / "final_model")
    UNLEARNED_PATH = str(
        MODEL_DIR / "unlearned_models" / "MaxEnt" / "pretrained_models_gemma-2-0.1B_all_arithmetic+eng_final_model_lr_8.0e-05" / "final_model"
    )

    # Hyperparameters
    percentile = 0.2
    EXCLUSIONS = ["self_attn", "layernorm", "embed_tokens"]

    mask_file_name = f"delta_mask_top_{int(percentile * 100)}_percent.pt"

    save_path = LOCALIZATION_MASKS_DIR / mask_file_name

    create_and_save_localization_mask(
        reference_model_path=PRETRAINED_PATH,
        unlearned_model_path=UNLEARNED_PATH,
        output_path=save_path,
        percentile=percentile,
        exclude_components=EXCLUSIONS
    )