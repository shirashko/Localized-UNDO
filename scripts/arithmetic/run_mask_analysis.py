import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from typing import Dict
from accelerate import Accelerator

from localized_undo.utils.paths import MODEL_DIR, DATASET_DIR, CACHE_DIR
from localized_undo.utils.mask_factory import MaskFactory
from localized_undo.utils.mask_ablation import StructuralAblationAnalyzer
from localized_undo.utils.validation_functions import get_arithmetic_eval_fn


def run_mask_diagnostic(
        original_model_path: str,
        unlearned_model_path: str,
        output_dir: str,
        percentile: float = 0.1,
        exclude_components: list = ["self_attn", "layernorm", "embed_tokens"]
):
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Accelerator once here
    accelerator = Accelerator()
    device = accelerator.device

    # Convert PosixPath to string for Hugging Face compatibility
    clean_eng_path = str(DATASET_DIR / "pretrain" / "valid_eng.jsonl")

    print(f"[*] Initializing evaluation function...")
    eval_fn = get_arithmetic_eval_fn(
        model_name=str(original_model_path),
        batch_size=8,
        max_length=256,
        num_wiki_batches=10,
        eng_valid_file=clean_eng_path,
        accelerator=accelerator,
        cache_dir=str(CACHE_DIR),
        dataset_cache_dir=str(CACHE_DIR)
    )

    # 2. Initialize the Ablation Analyzer with the shared accelerator
    analyzer = StructuralAblationAnalyzer(
        model_path=str(original_model_path),
        ref_model_path=str(unlearned_model_path),
        eval_fn=eval_fn,
        accelerator=accelerator,
        percentile=percentile,
        exclude_components=exclude_components
    )

    # 3. Explicit Mask Factory Diagnostic
    print(f"[*] Verifying Delta Mask via MaskFactory...")
    verified_mask = MaskFactory.get_mask(
        mask_type="delta_mask",
        model=analyzer.model,
        ref_model=analyzer.ref_model,
        percentile=percentile,
        exclude_components=exclude_components,
        device=device
    )

    mask_save_path = os.path.join(output_dir, "delta_mask.pt")
    print(f"[*] Saving Delta Mask to {mask_save_path}...")
    torch.save(verified_mask, mask_save_path)

    # 4. Run the Structural Comparison (Targeted vs. Random)
    print("\n[*] Starting Structural Ablation Comparison (Targeted vs. Random)...")
    ablation_results = analyzer.run_comparison(num_random_trials=3)

    # 5. Visualization & Metrics
    print("[*] Generating comparative plots...")
    analyzer.plot_results(ablation_results, os.path.join(output_dir, "ablation_comparison.png"))
    plot_mask_distribution(verified_mask, output_dir)

    # 6. Save results to JSON (Handling NumPy types for serialization)
    with open(os.path.join(output_dir, "ablation_metrics.json"), "w") as f:
        json.dump(
            ablation_results,
            f,
            indent=4,
            default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x
        )

    print(f"[+] Diagnostic complete. Results saved to {output_dir}")
    return ablation_results


def plot_mask_distribution(mask: Dict[str, torch.Tensor], output_dir: str):
    """Visualizes the percentage of masked parameters per layer."""
    layer_counts = {}
    for name, m in mask.items():
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            idx = int(match.group(1))
            if idx not in layer_counts:
                layer_counts[idx] = {"total": 0, "masked": 0}
            layer_counts[idx]["total"] += m.numel()
            layer_counts[idx]["masked"] += (m > 0).sum().item()

    layers = sorted(layer_counts.keys())
    densities = [layer_counts[l]["masked"] / layer_counts[l]["total"] for l in layers]

    plt.figure(figsize=(12, 6))
    plt.bar(layers, densities, color='teal', alpha=0.7, edgecolor='black')
    plt.title("Delta Mask Concentration per Layer", fontsize=14, fontweight='bold')
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Masked Weights Density", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    target_percentile = np.mean(densities) if densities else 0
    plt.axhline(y=target_percentile, color='red', linestyle='--', label=f'Avg Sparsity ({target_percentile:.4f})')

    plt.legend()
    plt.savefig(os.path.join(output_dir, "layer_distribution.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    pretrained = str(MODEL_DIR / "pretrained_models" / "gemma-2-0.1B_all_arithmetic+eng" / "final_model")
    unlearned = str(
        MODEL_DIR / "unlearned_models" / "MaxEnt" / "pretrained_models_gemma-2-0.1B_all_arithmetic+eng_final_model_lr_8.0e-05" / "final_model")

    run_mask_diagnostic(
        original_model_path=pretrained,
        unlearned_model_path=unlearned,
        output_dir="plots/diagnostics/ablation_study",
        percentile=0.2,
        exclude_components=["self_attn", "layernorm", "embed_tokens"]
    )