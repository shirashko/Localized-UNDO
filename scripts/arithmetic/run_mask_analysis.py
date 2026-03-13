import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM
from typing import Dict
from localized_undo.utils.paths import MODEL_DIR
from localized_undo.utils.mask_factory import MaskFactory

def run_mask_diagnostic(
        original_model_path: str,
        unlearned_model_path: str,
        output_dir: str,
        percentile: float = 0.1,
        exclude_components: list = ["self_attn", "layernorm", "embed_tokens"]
):
    """
    Diagnostic script to generate and visualize the Delta Mask properties.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    print(f"[*] Loading models for diagnostic...")
    model_orig = AutoModelForCausalLM.from_pretrained(original_model_path).to(device)
    model_unl = AutoModelForCausalLM.from_pretrained(unlearned_model_path).to(device)

    # 1. Generate the Delta Mask
    print(f"[*] Generating Delta Mask (Percentile: {percentile})...")
    mask = MaskFactory.get_mask(
        mask_type="delta_mask",
        model=model_unl,
        ref_model=model_orig,
        percentile=percentile,
        exclude_components=exclude_components,
        device=device
    )

    # 2. Analyze Mask Metrics
    print(f"[*] Analyzing mask distribution...")
    metrics = MaskFactory.analyze_mask(mask, model_unl)

    # 3. Structural Visualization (Layer-wise Density)
    plot_mask_distribution(mask, output_dir)

    # 4. Save results
    with open(os.path.join(output_dir, "mask_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[+] Diagnostic complete. Results saved to {output_dir}")
    return metrics


def plot_mask_distribution(mask: Dict[str, torch.Tensor], output_dir: str):
    """
    Creates a bar plot showing the density of masked weights per layer.
    """
    layer_counts = {}

    for name, m in mask.items():
        # Match layer index using regex
        import re
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
    plt.ylabel("Masked Weights Density (Sparsity)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Add a horizontal line for the target overall percentile
    target_percentile = np.mean(densities) if densities else 0
    plt.axhline(y=target_percentile, color='red', linestyle='--', label=f'Avg Sparsity ({target_percentile:.3f})')

    plt.legend()
    plt.savefig(os.path.join(output_dir, "layer_distribution.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    # Configuration for your specific experiment
    unlearned = MODEL_DIR / "pretrained_models" / "gemma-2-0.1B_all_arithmetic+eng" / "final_model"
    reference = MODEL_DIR / "unlearned_models" / "MaxEnt/pretrained_models_gemma-2-0.1B_all_arithmetic+eng_final_model_lr_8.0e-05" / "final_model"

    results = run_mask_diagnostic(
        original_model_path=reference,
        unlearned_model_path=unlearned,
        output_dir="plots/diagnostics/delta_analysis",
        percentile=0.1,  # Testing with 10% localization
    )
    print(f"Final Sparsity: {results['mask/total_sparsity']:.4f}")