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
    """
    Generates a full cross-product of masks:
    (Delta vs Random) x (Global vs Layer-wise) for each percentile.
    """
    print(f"[*] Starting Comprehensive Mask Sweep for percentiles: {percentiles}")
    device = torch.device("cpu")

    # Load Models Once for generation
    print(f"[*] Loading models to CPU for mask generation...")
    ref_model = AutoModelForCausalLM.from_pretrained(reference_model_path, torch_dtype=torch.float32, device_map="cpu")
    unlearned_model = AutoModelForCausalLM.from_pretrained(unlearned_model_path, torch_dtype=torch.float32,
                                                           device_map="cpu")

    created_folders = []

    for p in percentiles:
        print(f"\n--- Generating Masks for Percentile: {p} ---")

        # Define the modes we want to compare
        mask_types = ["delta_mask", "random"]
        distribution_modes = ["global", "layer-wise"]

        for dist_mode in distribution_modes:
            for m_type in mask_types:
                # Generate the specific mask configuration
                mask = MaskFactory.get_mask(
                    mask_type=m_type,
                    model=unlearned_model,
                    ref_model=ref_model if m_type == "delta_mask" else None,
                    percentile=p,
                    exclude_components=exclude_components,
                    device=device,
                    distribution_mode=dist_mode
                )

                # Run diagnostic analysis on the generated mask
                stats = MaskFactory.analyze_mask(mask)

                # Metadata for tracking
                metadata = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "config": {
                        "percentile": p,
                        "mask_type": m_type,
                        "distribution_mode": dist_mode,
                        "exclusions": exclude_components
                    },
                    "analysis": stats
                }

                p_int = int(p * 100) if p < 1.0 else int(p)
                folder_name = f"{base_folder_name}_p{p_int}_{m_type}_{dist_mode}"
                output_dir = LOCALIZATION_MASKS_DIR / folder_name
                os.makedirs(output_dir, exist_ok=True)

                # Save the individual mask and its metadata
                mask_filename = "mask.pt"  # Keep standard name for the diagnostic tool to find
                torch.save(mask, output_dir / mask_filename)

                with open(output_dir / "mask_config.json", "w") as f:
                    json.dump(metadata, f, indent=4)

                print(f"[+] Saved {m_type} ({dist_mode}) to: {output_dir}")
                created_folders.append(str(output_dir))

    return created_folders


def run_diagnostic_sweep(folders: list, model_to_test: str, eng_valid: str):
    """Applies the diagnostic suite to verify unlearning robustness for each mask mode."""
    print("\n" + "=" * 50)
    print("STARTING MECHANISTIC DIAGNOSTIC SWEEP")
    print("=" * 50)

    analyzer = MaskMechanisticDiagnostic(model_to_test, eng_valid)

    for folder in sorted(folders):
        try:
            print(f"[*] Analyzing folder: {os.path.basename(folder)}")
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
    # Step A: Create all mask variations
    sweep_folders = create_mask_sweep(
        reference_model_path=PRETRAINED_PATH,
        unlearned_model_path=UNLEARNED_PATH,
        percentiles=PERCENTILE_SWEEP,
        exclude_components=EXCLUSIONS
    )

    # Step B: Run diagnostics on the Unlearned model to test corruption effectiveness
    run_diagnostic_sweep(
        folders=sweep_folders,
        model_to_test=UNLEARNED_PATH,
        eng_valid=ENG_VALID
    )

    print("\n[SUCCESS] Generation and Diagnostic sweep complete.")