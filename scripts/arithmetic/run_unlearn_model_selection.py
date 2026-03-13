import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from localized_undo.utils.paths import CONFIG_DIR, PROJECT_ROOT
from localized_undo.utils.config_handler import load_unlearn_configs


def analyze_sweep_from_configs(yaml_path, base_setup_ids, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Analyzes model results using Relative Normalization, saves findings, and plots
    with non-overlapping x-axis labels.
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
                # 1. Normalized Language Preservation (Perplexity Ratio)
                lang_delta = final_val["val/eng_ce_loss"] - baseline["val/eng_ce_loss"]
                lang_preservation = math.exp(-max(lang_delta, 0))

                # 2. Relative Forget Efficiency (Multiplication & Division)
                forget_cols = [k for k in final_val.keys() if 'multiplication' in k or 'division' in k]
                base_f_acc = sum(baseline[k] for k in forget_cols) / len(forget_cols) if forget_cols else 1.0
                curr_f_acc = sum(final_val[k] for k in forget_cols) / len(forget_cols) if forget_cols else 0.0
                forget_efficiency = 1 - (curr_f_acc / max(base_f_acc, 0.001))

                # 3. Relative Math Retention (Addition & Subtraction)
                math_retain_cols = [k for k in final_val.keys() if 'addition' in k or 'subtraction' in k]
                base_m_acc = sum(baseline[k] for k in math_retain_cols) / len(
                    math_retain_cols) if math_retain_cols else 1.0
                curr_m_acc = sum(final_val[k] for k in math_retain_cols) / len(
                    math_retain_cols) if math_retain_cols else 0.0
                math_retention = curr_m_acc / max(base_m_acc, 0.001)

                all_final_metrics.append({
                    "setup_id": setup_id,
                    "lr": float(config['learning_rate']),
                    "lang_preserv_%": lang_preservation * 100,
                    "forget_eff_%": forget_efficiency * 100,
                    "math_retent_%": math_retention * 100,
                    "score": (alpha * lang_preservation) + (beta * forget_efficiency) + (gamma * math_retention),
                    "output_dir": config['output_dir']
                })

    if not all_final_metrics:
        print("No valid records found.")
        return None

    df = pd.DataFrame(all_final_metrics).sort_values(by="lr")
    best_model = df.loc[df['score'].idxmax()]
    results_out_dir = PROJECT_ROOT / "plots" / base_setup_ids[0] / "unlearning_sweep_results"
    results_out_dir.mkdir(parents=True, exist_ok=True)
    results_out_dir.mkdir(parents=True, exist_ok=True)

    # --- PLOTTING LOGIC ---
    plt.figure(figsize=(12, 7))
    plt.semilogx(df["lr"], df["lang_preserv_%"], marker='o', label="Language Preservation (General Knowledge)",
                 color="#2c7fb8", linewidth=2)
    plt.semilogx(df["lr"], df["forget_eff_%"], marker='s', label="Forget Efficiency (Multiplication & Division)",
                 color="#e31a1c", linewidth=2)
    plt.semilogx(df["lr"], df["math_retent_%"], marker='^', label="Math Retention (Addition & Subtraction)",
                 color="#31a354", linewidth=2)

    # Highlight Best
    plt.axvline(x=best_model['lr'], color='orange', linestyle='--', alpha=0.6,
                label=f"Best Model (LR={best_model['lr']:.1e})")

    # Clean Title
    clean_title = base_setup_ids[0].replace("_", " ")
    plt.title(f"Unlearning Sweep Analysis: {clean_title}\nNormalized Trade-off Metrics", fontsize=14)
    plt.xlabel("Learning Rate (Log Scale)", fontsize=12)
    plt.ylabel("Relative Performance (%)", fontsize=12)

    # FIX: Smaller font size and rotation for x-axis ticks to prevent overlap
    plt.xticks(df["lr"], [f"{lr:.1e}" for lr in df["lr"]], fontsize=9, rotation=30)

    plt.legend(loc='lower left')
    plt.grid(True, which="both", linestyle="--", alpha=0.3)

    plot_path = results_out_dir / f"sweep_plot_{base_setup_ids[0]}.pdf"
    plt.savefig(plot_path, format='pdf', bbox_inches='tight')
    print(f"📈 Plot saved to: {plot_path}")

    # --- SAVING DATA ---
    csv_path = results_out_dir / f"unlearn_sweep_{base_setup_ids[0]}.csv"
    df.sort_values(by="score", ascending=False).to_csv(csv_path, index=False)

    best_json_path = results_out_dir / "best_unlearned_checkpoint.json"
    with open(best_json_path, 'w') as f:
        json.dump(best_model.to_dict(), f, indent=4)

    return best_model


if __name__ == "__main__":
    BASE_SETUPS = ["gemma-2-0.1B_MaxEnt"]
    YAML_PATH = CONFIG_DIR / "arithmetic" / "unlearn.yaml"
    analyze_sweep_from_configs(YAML_PATH, BASE_SETUPS)