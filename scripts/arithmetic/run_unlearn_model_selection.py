import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from localized_undo.utils.paths import CONFIG_DIR, PROJECT_ROOT
from localized_undo.utils.config_handler import load_unlearn_configs


def analyze_sweep_from_configs(yaml_path, base_setup_ids, forget_weight=1.0, retain_weight=1.0):
    """
    Analyzes unlearning sweeps using two evaluation modes:
    1. Relative Mode: Performance compared to the pretrained baseline.
    2. Absolute Mode: Raw performance at the final step (end state only).
    """
    all_exp_configs = load_unlearn_configs(yaml_path, base_setup_ids)
    all_final_metrics = []

    for setup_id, config in all_exp_configs.items():
        record_path = Path(config['path_local_record'])
        if not record_path.exists():
            continue

        with open(record_path, 'r') as f:
            lines = f.readlines()
            if not lines: continue

            baseline, final_val = None, None
            for line in lines:
                data = json.loads(line)
                if "val/eng_ce_loss" in data:
                    if baseline is None: baseline = data
                    final_val = data

            if final_val and baseline:
                # Identify relevant metric columns
                forget_cols = [k for k in final_val.keys() if 'multiplication' in k or 'division' in k]
                math_retain_cols = [k for k in final_val.keys() if 'addition' in k or 'subtraction' in k]

                # --- CALCULATE RAW ACCURACIES (Absolute Values) ---
                current_forget_acc = np.mean([final_val[k] for k in forget_cols]) if forget_cols else 0.0
                current_retain_acc = np.mean([final_val[k] for k in math_retain_cols]) if math_retain_cols else 0.0

                # --- MODE 1: RELATIVE METRICS (Efficiency vs Baseline) ---
                base_forget_acc = np.mean([baseline[k] for k in forget_cols]) if forget_cols else 1.0
                base_retain_acc = np.mean([baseline[k] for k in math_retain_cols]) if math_retain_cols else 1.0

                rel_forget_efficiency = 1 - (current_forget_acc / max(base_forget_acc, 0.001))
                rel_math_retention = current_retain_acc / max(base_retain_acc, 0.001)
                relative_score = (forget_weight * rel_forget_efficiency) + (retain_weight * rel_math_retention)

                # --- MODE 2: ABSOLUTE METRICS (Final State Only) ---
                # For forgetting: Score is 1.0 if accuracy is 0.0 (Perfect erasure)
                abs_forget_score = 1 - current_forget_acc
                # For retention: Score is 1.0 if accuracy is 1.0 (Perfect preservation)
                abs_retain_score = current_retain_acc
                absolute_score = (forget_weight * abs_forget_score) + (retain_weight * abs_retain_score)

                all_final_metrics.append({
                    "setup_id": setup_id,
                    "lr": float(config['learning_rate']),
                    # Relative Data
                    "rel_forget_eff_%": rel_forget_efficiency * 100,
                    "rel_math_retent_%": rel_math_retention * 100,
                    "relative_composite_score": relative_score,
                    # Absolute Data
                    "abs_forget_acc_%": current_forget_acc * 100,
                    "abs_retain_acc_%": current_retain_acc * 100,
                    "absolute_composite_score": absolute_score,
                    "output_dir": config['output_dir']
                })

    if not all_final_metrics:
        print("[!] No valid records found.")
        return None

    df = pd.DataFrame(all_final_metrics).sort_values(by="lr")
    results_out_dir = PROJECT_ROOT / "plots" / base_setup_ids[0] / "unlearning_sweep_results"
    results_out_dir.mkdir(parents=True, exist_ok=True)

    # --- PLOT 1: RELATIVE EFFICIENCY (Baseline Comparison) ---
    plt.figure(figsize=(10, 6))
    best_rel_model = df.loc[df['relative_composite_score'].idxmax()]
    plt.semilogx(df["lr"], df["rel_forget_eff_%"], marker='s', label="Forget Efficiency (vs Baseline)", color="#e31a1c")
    plt.semilogx(df["lr"], df["rel_math_retent_%"], marker='^', label="Math Retention (vs Baseline)", color="#31a354")
    plt.axvline(x=best_rel_model['lr'], color='orange', linestyle='--',
                label=f"Best Rel LR: {best_rel_model['lr']:.1e}")
    plt.title(f"Option 1: Relative Unlearning Efficiency\n(Compared to Baseline Model)")
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Relative Change (%)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(results_out_dir / f"sweep_relative_{base_setup_ids[0]}.pdf", bbox_inches='tight')

    # --- PLOT 2: ABSOLUTE PERFORMANCE (Final State Only) ---
    plt.figure(figsize=(10, 6))
    best_abs_model = df.loc[df['absolute_composite_score'].idxmax()]
    plt.semilogx(df["lr"], df["abs_forget_acc_%"], marker='o', label="Final Forget Accuracy (Goal: 0%)",
                 color="#bcbddc")
    plt.semilogx(df["lr"], df["abs_retain_acc_%"], marker='d', label="Final Retain Accuracy (Goal: 100%)",
                 color="#756bb1")
    plt.axvline(x=best_abs_model['lr'], color='purple', linestyle='--',
                label=f"Best Abs LR: {best_abs_model['lr']:.1e}")
    plt.title(f"Option 2: Absolute Unlearning Performance\n(Final Model State Only)")
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Raw Accuracy (%)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(results_out_dir / f"sweep_absolute_{base_setup_ids[0]}.pdf", bbox_inches='tight')

    # Save comprehensive CSV
    df.to_csv(results_out_dir / f"unlearn_sweep_dual_mode_{base_setup_ids[0]}.csv", index=False)
    print(f"[+] Analysis complete. Two plots saved to: {results_out_dir}")

    return best_rel_model, best_abs_model


if __name__ == "__main__":
    BASE_SETUPS = ["gemma-2-0.1B_MaxEnt"]
    YAML_PATH = CONFIG_DIR / "arithmetic" / "unlearn.yaml"
    analyze_sweep_from_configs(YAML_PATH, BASE_SETUPS)