import os
import torch
import json
from datetime import datetime
from transformers import AutoModelForCausalLM
from localized_undo.utils.paths import MODEL_DIR, LOCALIZATION_MASKS_DIR, DATASET_DIR
from localized_undo.utils.mask_factory import MaskFactory
from localized_undo.utils.localization_diagnoser import MaskMechanisticDiagnostic


def create_mask_sweep(
        reference_model_path: str,
        unlearned_model_path: str,
        percentiles: list[float],
        exclude_components: list = ["self_attn", "layernorm", "embed_tokens"],
        base_folder_name: str = "arithmetic"
):
    """Generates paired Delta and Random masks for a sweep of percentiles."""
    print(f"[*] Starting Mask Sweep for percentiles: {percentiles}")
    device = torch.device("cpu")

    # Load Models Once for generation
    print(f"[*] Loading models to CPU for mask generation...")
    ref_model = AutoModelForCausalLM.from_pretrained(reference_model_path, torch_dtype=torch.float32, device_map="cpu")
    unlearned_model = AutoModelForCausalLM.from_pretrained(unlearned_model_path, torch_dtype=torch.float32,
                                                           device_map="cpu")

    created_folders = []

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

        delta_stats = MaskFactory.analyze_mask(mask_results["delta_mask"], unlearned_model)

        # Metadata logic
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {"percentile": p, "exclusions": exclude_components},
            "sparsity": f"{delta_stats['mask/total_sparsity']:.4%}"
        }

        p_int = int(p * 100) if p < 1.0 else int(p)
        folder_name = f"{base_folder_name}_p{p_int}_excl_{'_'.join(exclude_components)}"
        output_dir = LOCALIZATION_MASKS_DIR / folder_name
        os.makedirs(output_dir, exist_ok=True)

        torch.save(mask_results["delta_mask"], output_dir / "delta_mask.pt")
        torch.save(mask_results["random"], output_dir / "random_baseline.pt")
        with open(output_dir / "mask_config.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[+] Saved percentile {p} to: {output_dir}")
        created_folders.append(str(output_dir))

    return created_folders


def run_diagnostic_sweep(folders: list, model_to_test: str, eng_valid: str):
    """Applies the diagnostic suite to a list of folders using the Unlearned model."""
    print("\n" + "=" * 50)
    print("STARTING MECHANISTIC DIAGNOSTIC SWEEP")
    print("=" * 50)

    analyzer = MaskMechanisticDiagnostic(model_to_test, eng_valid)

    for folder in sorted(folders):
        try:
            analyzer.run_on_folder(folder)
        except Exception as e:
            print(f"[!] Error analyzing {folder}: {e}")


if __name__ == "__main__":
    # 1. Setup Paths
    PRETRAINED_PATH = str(MODEL_DIR / "pretrained_models" / "gemma-2-0.1B_all_arithmetic+eng" / "final_model")
    UNLEARNED_PATH = str(
        MODEL_DIR / "unlearned_models" / "MaxEnt" / "pretrained_models_gemma-2-0.1B_all_arithmetic+eng_final_model_lr_8.0e-05" / "final_model")
    ENG_VALID = str(DATASET_DIR / "pretrain" / "valid_eng.jsonl")

    # 2. Setup Sweep Params
    PERCENTILE_SWEEP = [0.05, 0.1, 0.2, 0.3, 0.5]
    EXCLUSIONS = ["self_attn", "layernorm", "embed_tokens"]

    # 3. Execute
    # Step A: Create the masks
    sweep_folders = create_mask_sweep(
        reference_model_path=PRETRAINED_PATH,
        unlearned_model_path=UNLEARNED_PATH,
        percentiles=PERCENTILE_SWEEP,
        exclude_components=EXCLUSIONS
    )

    # Step B: Run diagnostics on the Unlearned model
    run_diagnostic_sweep(
        folders=sweep_folders,
        model_to_test=UNLEARNED_PATH,  # Testing on the model we want to "fix"
        eng_valid=ENG_VALID
    )

    print("\n[SUCCESS] Generation and Diagnostic sweep complete.")