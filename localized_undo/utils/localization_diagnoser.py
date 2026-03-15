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
        Diagnostic suite to audit unlearning masks.
        Tests the mask by applying it to the model (usually the Unlearned one)
        and measuring the performance diff.
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

        # Cache original weights to restore them between Targeted and Random tests
        self.original_weights = {n: p.data.clone() for n, p in self.model.named_parameters()}

        # Initialize arithmetic evaluation suite
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
        """Resets model to the baseline state (the weights at initialization)."""
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.data.copy_(self.original_weights[n])

    @torch.no_grad()
    def apply_mask_erasure(self, mask: Dict[str, torch.Tensor]):
        """Intervention: Zeros out weights using standardized project naming."""
        applied_layers = 0

        for n, p in self.model.named_parameters():
            # Standardizing name using the shared utility
            clean_n = clean_parameter_name(n)

            if clean_n in mask:
                m = mask[clean_n].to(self.device)

                # Apply erasure: mask == 1.0 means we zero the weight
                p.data *= (1.0 - m.view(p.shape))
                applied_layers += 1

        print(f"[DEBUG] Mask applied to {applied_layers} layers.")
        if applied_layers == 0:
            # Critical error check for naming mismatches
            model_keys = list(self.model.state_dict().keys())[:3]
            mask_keys = list(mask.keys())[:3]
            print(f"[!] ERROR: Zero layers were modified. Check naming mismatch!")
            print(f"    Sample Model Keys: {model_keys}")
            print(f"    Sample Mask Keys:  {mask_keys}")

    def run_on_folder(self, folder_path):
        """Processes a single configuration folder (e.g., p50_delta_global)."""
        results = {}
        loc = str(self.device)

        # Every mask configuration is saved as mask.pt in its unique dir
        mask_path = os.path.join(folder_path, "mask.pt")

        if not os.path.exists(mask_path):
            print(f"[!] Skipping {folder_path}: mask.pt not found.")
            return None

        print(f"\n[Diagnostic] Analyzing: {os.path.basename(folder_path)}")

        # 1. Baseline: Evaluate the Unlearned model as is
        self.restore()
        results["Baseline_Unlearned"] = self.eval_fn(self.model, print_results=False)

        # 2. Intervention: Apply the mask and evaluate
        self.restore()  # Ensure we start from a clean state
        self.apply_mask_erasure(torch.load(mask_path, map_location=loc))
        results["Mask_Applied"] = self.eval_fn(self.model, print_results=False)

        # 3. Final Restoration
        self.restore()

        # GPU Maintenance
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Save Metrics
        with open(os.path.join(folder_path, "diagnostic_metrics.json"), "w") as f:
            json.dump(results, f, indent=4)

        # Save Plots
        self.plot_diagnostic(results, folder_path)

        return results

    def plot_diagnostic(self, results, output_dir):
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
        ax1.set_xticklabels(labels)
        ax1.set_title("Erasure impact on Arithmetic Tasks")
        ax1.legend()

        ax2.bar(labels, eng_losses, color=['gray', 'blue'])
        ax2.set_title("Language Damage (CE Loss)")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "localization_diagnostic.png"))
        plt.close()