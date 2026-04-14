from collections import defaultdict
from datetime import datetime
import json
import statistics


def _infer_method_name(setup_id: str) -> str:
    if "_MaxEnt" in setup_id:
        return "MaxEnt"
    if "_RMU" in setup_id:
        return "RMU"
    if "_GradDiff" in setup_id:
        return "GradDiff"
    if "_SAM" in setup_id:
        return "SAM"
    if "_repnoise" in setup_id:
        return "repnoise"
    return "Unknown"


def _extract_last_metric(record_path: str, task_prefix: str, shots_suffix: str):
    last_value = None
    try:
        with open(record_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key, value in row.items():
                    if key.startswith(task_prefix) and key.endswith(shots_suffix):
                        try:
                            last_value = float(value)
                        except (TypeError, ValueError):
                            pass
    except FileNotFoundError:
        return None
    return last_value


def write_summary_log(setups, setups_to_run, summary_path="scripts/wmdp/logs/wmdp_unlearn_summary.log"):
    grouped = defaultdict(lambda: {"wmdp": [], "mmlu": []})
    for setup_id in setups_to_run:
        cfg = setups[setup_id]
        method = _infer_method_name(setup_id)
        alpha = cfg["alpha"]
        record_path = cfg["path_local_record"]

        wmdp_val = _extract_last_metric(
            record_path=record_path,
            task_prefix="wmdp_bio_limit_",
            shots_suffix="_shots_0",
        )
        mmlu_val = _extract_last_metric(
            record_path=record_path,
            task_prefix="mmlu_limit_",
            shots_suffix="_shots_5",
        )
        if wmdp_val is not None:
            grouped[(method, alpha)]["wmdp"].append(wmdp_val)
        if mmlu_val is not None:
            grouped[(method, alpha)]["mmlu"].append(mmlu_val)

    lines = []
    lines.append(
        "method\talpha\tWMDP-Bio mean Accuracy (Forget ↓)\tWMDP-Bio std Accuracy (Forget ↓)\tMMLU mean Accuracy (Retain ↑)\tMMLU std Accuracy (Retain ↑)"
    )
    for (method, alpha), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        wmdp_scores = vals["wmdp"]
        mmlu_scores = vals["mmlu"]
        wmdp_mean = statistics.mean(wmdp_scores) if wmdp_scores else float("nan")
        mmlu_mean = statistics.mean(mmlu_scores) if mmlu_scores else float("nan")
        wmdp_std = statistics.stdev(wmdp_scores) if len(wmdp_scores) > 1 else 0.0
        mmlu_std = statistics.stdev(mmlu_scores) if len(mmlu_scores) > 1 else 0.0
        lines.append(f"{method}\t{alpha}\t{wmdp_mean:.4f}\t{wmdp_std:.4f}\t{mmlu_mean:.4f}\t{mmlu_std:.4f}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== WMDP Unlearn Summary @ {timestamp} ===\n")
        for line in lines:
            f.write(line + "\n")

    print("\n=== WMDP Unlearn Summary ===")
    for line in lines:
        print(line)
    print(f"\nSummary appended to: {summary_path}")
