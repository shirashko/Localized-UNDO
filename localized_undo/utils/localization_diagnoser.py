import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from localized_undo.utils.validation_functions import get_arithmetic_eval_fn


class MaskMechanisticDiagnostic:
    def __init__(self, model_path: str, eng_valid_path: str, device="cpu"):
        """
        Diagnostic suite to audit unlearning masks.
        Tests the mask by applying it to the model (usually the Unlearned one)
        and measuring the delta in performance.
        """
        self.device = torch.device(device)
        self.accelerator = Accelerator(cpu=(device == "cpu"))
        self.model_path = model_path

        print(f"[*] Loading model for diagnostics: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

        # Cache original weights to restore them between Targeted and Random tests
        self.original_weights = {n: p.data.clone() for n, p in self.model.named_parameters()}

        # Initialize arithmetic evaluation suite
        self.eval_fn = get_arithmetic_eval_fn(
            model_name=model_path, batch_size=8, max_length=256,
            num_wiki_batches=50, eng_valid_file=eng_valid_path,
            accelerator=self.accelerator
        )

    def restore(self):
        """Resets model to the state it was in at init (the baseline unlearned state)."""
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                p.data.copy_(self.original_weights[n])

    @torch.no_grad()
    def apply_mask_erasure(self, mask: Dict[str, torch.Tensor]):
        """Intervention: Zeros out weights indicated by the binary mask."""
        for n, p in self.model.named_parameters():
            # Standard cleanup to match mask keys
            clean_n = n.replace("model.", "").replace("module.", "")
            if clean_n in mask:
                m = mask[clean_n].to(self.device)
                # mask == 1 means "top discrepancy", so we erase it (multiply by 0)
                p.data *= (1.0 - m.view(p.shape))

    def run_on_folder(self, folder_path):
        """Processes a single sweep folder."""
        results = {}
        delta_path = os.path.join(folder_path, "delta_mask.pt")
        random_path = os.path.join(folder_path, "random_baseline.pt")

        if not os.path.exists(delta_path):
            print(f"[!] Skipping {folder_path}: delta_mask.pt not found.")
            return None

        print(f"\n[Diagnostic] Analyzing: {os.path.basename(folder_path)}")

        # 1. Baseline: Performance of the Unlearned model as-is
        print("[*] Evaluating Baseline (Unlearned state)...")
        self.restore()
        results["Baseline_Unlearned"] = self.eval_fn(self.model, print_results=False)

        # 2. Targeted: Apply Delta Mask erasure to the Unlearned model
        print("[*] Evaluating Targeted Erasure (Delta Mask)...")
        self.restore()
        self.apply_mask_erasure(torch.load(delta_path, map_location='cpu'))
        results["Delta_Targeted"] = self.eval_fn(self.model, print_results=False)

        # 3. Control: Apply Random Mask baseline
        if os.path.exists(random_path):
            print("[*] Evaluating Random Control (Baseline)...")
            self.restore()
            self.apply_mask_erasure(torch.load(random_path, map_location='cpu'))
            results["Random_Control"] = self.eval_fn(self.model, print_results=False)

        self.restore()  # Cleanup

        # Save Metrics and Plots
        with open(os.path.join(folder_path, "diagnostic_metrics.json"), "w") as f:
            json.dump(results, f, indent=4)

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
        ax1.set_xticks(x);
        ax1.set_xticklabels(labels)
        ax1.set_title("Erasure impact on Arithmetic Tasks")
        ax1.legend()

        ax2.bar(labels, eng_losses, color=['gray', 'blue', 'lightblue'][:len(labels)])
        ax2.set_title("Language Damage (CE Loss)")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "localization_diagnostic.png"))
        plt.close()