import argparse
import json

from accelerate import Accelerator
from transformers import AutoModelForCausalLM

from localized_undo.utils.loss_functions import custom_login
from localized_undo.utils.paths import CACHE_DIR, WMDP_MODEL_DIR
from localized_undo.utils.validation_functions import (
    eval_model_lm_eval,
    get_both_wmdp_eval_fn,
    get_wmdp_bio_eval_fn,
    get_wmdp_cyber_eval_fn,
)


def _load_model(model_name: str, cache_dir: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        attn_implementation="eager",
    )
    return model


def _compute_mmlu_bio_breakdown(model, accelerator: Accelerator, large_eval: bool, full_mmlu: bool):
    lim = None if full_mmlu else (0.40 if large_eval else 0.07)
    seed = 1234 if large_eval else None

    # Full MMLU subtask results (e.g., mmlu_college_biology_limit_0.4_shots_5)
    subtask_results = eval_model_lm_eval(
        model=model,
        print_results=False,
        accelerator=accelerator,
        seed=seed,
        task_list=["mmlu"],
        limit=[lim],
        keep_all_subtasks=True,
    )

    bio_keywords = (
        "biology",
        "anatomy",
        "medical",
        "medicine",
        "clinical",
        "aging",
        "virology",
        "nutrition",
        "genetics",
    )
    time_suffix = " time"

    bio_vals = []
    non_bio_vals = []
    for key, value in subtask_results.items():
        if key.endswith(time_suffix):
            continue
        if "mmlu_" not in key:
            continue
        if any(keyword in key for keyword in bio_keywords):
            bio_vals.append(value)
        else:
            non_bio_vals.append(value)

    out = {}
    if bio_vals:
        out["mmlu_bio_related_avg"] = sum(bio_vals) / len(bio_vals)
        out["mmlu_bio_related_count"] = len(bio_vals)
    if non_bio_vals:
        out["mmlu_non_bio_related_avg"] = sum(non_bio_vals) / len(non_bio_vals)
        out["mmlu_non_bio_related_count"] = len(non_bio_vals)
    return out


def _task_list_and_limits(domain: str, large_eval: bool, no_mmlu: bool, full_mmlu: bool):
    mmlu_lim = None if full_mmlu else (0.40 if large_eval else 0.07)
    if domain == "bio":
        if no_mmlu:
            return ["wmdp_bio"], [None] if large_eval else [1000]
        return ["wmdp_bio", "mmlu"], [None, mmlu_lim] if large_eval else [1000, mmlu_lim]
    if domain == "cyber":
        if no_mmlu:
            return ["wmdp_cyber"], [None] if large_eval else [1000]
        return ["wmdp_cyber", "mmlu"], [None, mmlu_lim] if large_eval else [1000, mmlu_lim]

    if no_mmlu:
        raise ValueError("--no-mmlu is not supported with --domain both")
    return ["wmdp_bio", "wmdp_cyber", "mmlu"], [None, None, mmlu_lim] if large_eval else [1000, 1000, mmlu_lim]

def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation (WMDP/MMLU) on a model checkpoint.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=f"{WMDP_MODEL_DIR}/gemma-2-2b",
        help="Model path or HF model id to evaluate.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["bio", "cyber", "both"],
        default="bio",
        help="Which WMDP domain eval function to run.",
    )
    parser.add_argument(
        "--large-eval",
        action="store_true",
        help="Use large eval settings (same meaning as in run_unlearn_wmdp.py).",
    )
    parser.add_argument(
        "--no-mmlu",
        action="store_true",
        help="Disable MMLU evaluation where supported.",
    )
    parser.add_argument(
        "--report-mmlu-bio-breakdown",
        action="store_true",
        help="Additionally report MMLU biology-related vs non-biology-related averages.",
    )
    parser.add_argument(
        "--report-all-subtasks",
        action="store_true",
        help="Additionally report all WMDP/MMLU subtasks (keep_all_subtasks=True).",
    )
    parser.add_argument(
        "--full-mmlu",
        action="store_true",
        help="Evaluate full MMLU (no limit).",
    )
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR), help="HF cache directory.")
    args = parser.parse_args()

    if args.report_mmlu_bio_breakdown and args.no_mmlu:
        raise ValueError("--report-mmlu-bio-breakdown requires MMLU. Remove --no-mmlu.")

    custom_login()
    accelerator = Accelerator()
    model = _load_model(args.model_name, args.cache_dir)
    model = accelerator.prepare(model)

    if args.full_mmlu:
        task_list, limits = _task_list_and_limits(
            domain=args.domain,
            large_eval=args.large_eval,
            no_mmlu=args.no_mmlu,
            full_mmlu=True,
        )
        seed = 1234 if args.large_eval else None
        eval_results = eval_model_lm_eval(
            model=model,
            print_results=True,
            accelerator=accelerator,
            seed=seed,
            task_list=task_list,
            limit=limits,
            keep_all_subtasks=False,
        )
    else:
        if args.domain == "bio":
            eval_fn = get_wmdp_bio_eval_fn(
                accelerator=accelerator,
                large_eval=args.large_eval,
                no_mmlu=args.no_mmlu,
            )
        elif args.domain == "cyber":
            eval_fn = get_wmdp_cyber_eval_fn(
                accelerator=accelerator,
                large_eval=args.large_eval,
                no_mmlu=args.no_mmlu,
            )
        else:
            if args.no_mmlu:
                raise ValueError("--no-mmlu is not supported with --domain both")
            eval_fn = get_both_wmdp_eval_fn(accelerator=accelerator, large_eval=args.large_eval)

        eval_results = eval_fn(model, print_results=True)
    if args.report_all_subtasks:
        task_list, limits = _task_list_and_limits(
            domain=args.domain,
            large_eval=args.large_eval,
            no_mmlu=args.no_mmlu,
            full_mmlu=args.full_mmlu,
        )
        seed = 1234 if args.large_eval else None
        all_subtasks = eval_model_lm_eval(
            model=model,
            print_results=False,
            accelerator=accelerator,
            seed=seed,
            task_list=task_list,
            limit=limits,
            keep_all_subtasks=True,
        )
        eval_results.update(all_subtasks)
    if args.report_mmlu_bio_breakdown:
        breakdown = _compute_mmlu_bio_breakdown(
            model=model,
            accelerator=accelerator,
            large_eval=args.large_eval,
            full_mmlu=args.full_mmlu,
        )
        eval_results.update(breakdown)
    if accelerator.is_main_process:
        print("\n=== Benchmark Eval Results ===")
        print(json.dumps(eval_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
