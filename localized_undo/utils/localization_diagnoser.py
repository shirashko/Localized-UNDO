import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from localized_undo.utils.validation_functions import get_arithmetic_eval_fn
from localized_undo.utils.paths import CACHE_DIR
from localized_undo.utils.localization_utils import clean_parameter_name


class MaskMechanisticDiagnostic:
    def __init__(self, model_path: str, eng_valid_path: str, device="cpu"):
        """
        Diagnostic suite to audit unlearning masks using Corruption (Shrink & Perturb).
        """
        self.device = torch.device(device)
        self.accelerator = Accelerator(cpu=(self.device.type == "cpu"))
        self.model_path = model_path

        print(f"[*] Loading model for diagnostics: {model_path} on {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

        # Cache original weights to restore them between tests
        self.original_weights = {n: p.data.clone() for n, p in self.model.named_parameters()}

        self.eval_fn = get_arithmetic_eval_fn(
            model_name=model_path,
            batch_size=8,
            max_length=256,
            num_wiki_batches=50,
            eng_valid_file=eng_valid_path,
            accelerator=self.accelerator,
            dataset_cache_dir=str(CACHE_DIR),
            cache_dir=str(CACHE_DIR),
        )

    def restore(self):
        """Resets model to the baseline state."""
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.data.copy_(self.original_weights[n])

    @torch.no_grad()
    def apply_mask_corruption(self, mask: Dict[str, torch.Tensor], noise_alpha: float, noise_beta: float,
                              seed: int = 42):
        """
        Intervention: Applies 'Shrink and Perturb' corruption instead of zeroing out.
        Formula: W_new = (1 - alpha*mask) * W_old + (alpha*mask) * (noise_beta * Noise)
        """
        torch.manual_seed(seed)
        applied_layers = 0

        # Pre-clean mask keys to match model naming
        normalized_mask = {k.replace("model.", ""): v for k, v in mask.items()}

        for n, p in self.model.named_parameters():
            clean_n = clean_parameter_name(n)

            # Handle standard naming cleanup similar to do_corruption
            if clean_n.startswith("model."):
                clean_n = clean_n[6:]

            if clean_n in normalized_mask:
                m = normalized_mask[clean_n].to(device=p.device, dtype=p.dtype)

                # Generate Noise (Xavier Uniform for weights, zero for others)
                if len(p.data.shape) >= 2:
                    noise = torch.nn.init.xavier_uniform_(torch.empty_like(p.data))
                else:
                    noise = torch.zeros_like(p.data)

                corruption = noise_beta * noise
                effective_alpha = noise_alpha * m.view(p.shape)

                # Apply Corruption Formula
                p.data = (1.0 - effective_alpha) * p.data + effective_alpha * corruption
                applied_layers += 1

        print(f"[DEBUG] Corruption applied to {applied_layers} layers using alpha={noise_alpha}, beta={noise_beta}.")

    def run_on_folder(self, folder_path, noise_alpha: float = 0.1, noise_beta: float = 0.1):
        """Processes a folder and evaluates the corrupted model."""
        results = {}
        mask_path = os.path.join(folder_path, "mask.pt")

        if not os.path.exists(mask_path):
            return None

        print(f"\n[Diagnostic] Analyzing: {os.path.basename(folder_path)} with Corruption")

        # 1. Baseline
        self.restore()
        results["Baseline_Unlearned"] = self.eval_fn(self.model, print_results=False)

        # 2. Intervention: Apply Corruption instead of erasure
        self.restore()
        mask = torch.load(mask_path, map_location=self.device)
        self.apply_mask_corruption(mask, noise_alpha=noise_alpha, noise_beta=noise_beta)
        results[f"Corrupted_a{noise_alpha}_b{noise_beta}"] = self.eval_fn(self.model, print_results=False)

        self.restore()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        with open(os.path.join(folder_path, "diagnostic_corruption_metrics.json"), "w") as f:
            json.dump(results, f, indent=4)

        self.plot_diagnostic(results, folder_path, noise_alpha, noise_beta)
        return results

    def plot_diagnostic(self, results, output_dir, alpha, beta):
        forget_keys = ['val/multiplication_equation_acc', 'val/division_equation_acc']
        retain_keys = ['val/addition_equation_acc', 'val/subtraction_equation_acc']

        labels = list(results.keys())
        forget_accs = [np.mean([results[l].get(k, 0) for k in forget_keys]) for l in labels]
        retain_accs = [np.mean([results[l].get(k, 0) for k in retain_keys]) for l in labels]
        eng_losses = [results[l].get('val/eng_ce_loss', 0) for l in labels]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        x = np.arange(len(labels))

        ax1.bar(x - 0.2, forget_accs, 0.4, label='Forget Set Acc', color='crimson')
        ax1.bar(x + 0.2, retain_accs, 0.4, label='Retain Set Acc', color='seagreen')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15)
        ax1.set_title(f"Corruption Impact (Alpha={alpha}, Beta={beta}) on Arithmetic Tasks")
        ax1.legend()

        ax2.bar(labels, eng_losses, color=['gray', 'blue'])
        ax2.set_title("Language Damage (CE Loss)")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "localization_corruption_diagnostic.png"))
        plt.close()