import argparse
import sys
from datetime import datetime
from accelerate import Accelerator

# Tooling and Utils
from localized_undo.tools.partial_distill_langarith import partial_distill
from localized_undo.utils.paths import CONFIG_DIR
from localized_undo.utils.config_handler import load_distill_configs
from localized_undo.utils.loss_functions import custom_login
from localized_undo.utils.validation_functions import get_arithmetic_eval_fn
from localized_undo.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

# Authenticate with HF and WandB globally
custom_login()


def arithmetic_stop_cond_fn(student_eval_dict, teacher_eval_dict, config):
    """
    Implements research-grade early stopping logic based on thresholds from YAML.
    Tracks relative English CE loss and absolute Arithmetic accuracy changes.
    """
    # 1. English retention (relative loss increase)
    student_eng_loss = student_eval_dict.get("val/eng_ce_loss", float('inf'))
    teacher_eng_loss = teacher_eval_dict.get("val/eng_ce_loss", 0)
    eng_diff = (student_eng_loss - teacher_eng_loss) / max(teacher_eng_loss, 1e-6)

    # 2. Arithmetic evaluation (Retain: Add/Sub, Forget: Mul/Div)
    retain_ops, forget_ops = ["addition", "subtraction"], ["multiplication", "division"]
    retain_metrics, forget_metrics = [], []

    for op in retain_ops + forget_ops:
        for fmt in ["equation", "word_problem"]:
            key = f"val/{op}_{fmt}_acc"
            # Difference: student_acc - teacher_acc (negative means student is worse)
            diff = student_eval_dict.get(key, 0) - teacher_eval_dict.get(key, 1.0)
            if op in retain_ops:
                retain_metrics.append(diff)
            else:
                forget_metrics.append(diff)

    avg_retain_diff = sum(retain_metrics) / max(len(retain_metrics), 1)
    avg_forget_diff = sum(forget_metrics) / max(len(forget_metrics), 1)

    # 3. Threshold checks against YAML-loaded criteria
    cond_eng = config['english_threshold'] is None or eng_diff < config['english_threshold']
    cond_retain = config['retain_arithmetic_threshold'] is None or avg_retain_diff > -config[
        'retain_arithmetic_threshold']
    cond_forget = config['forget_arithmetic_threshold'] is None or avg_forget_diff < -config[
        'forget_arithmetic_threshold']

    # 4. Selection Logic based on the 'stop_condition' key
    mode = config['stop_condition']
    if mode == "english_only": return cond_eng
    if mode == "retain_arithmetic_only": return cond_retain
    if mode == "forget_arithmetic_only": return cond_forget
    return cond_eng and cond_retain and cond_forget


def launch_worker(exp_id, all_configs):
    """
    Target function for parallel GPU execution.
    Each GPU runs a separate process for a unique Alpha/Beta/Seed combination.
    """
    config = all_configs[exp_id]
    accelerator = Accelerator()

    # Re-authenticate in the worker process for multi-GPU stability
    custom_login()

    # Initialize specialized arithmetic evaluation function
    eval_fn = get_arithmetic_eval_fn(
        model_name=config['student_model_name'],
        eng_valid_file=config['eng_valid_file'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        cache_dir=config['cache_dir'],
        dataset_cache_dir=config['dataset_cache_dir'],
        num_wiki_batches=50,
        accelerator=accelerator
    )

    # Wrap the stop condition to inject the current experiment's config
    def stop_wrapper(s_dict, t_dict):
        return arithmetic_stop_cond_fn(s_dict, t_dict, config)

    # Filter keys that are NOT defined in partial_distill's signature
    exclude = {
        'method', 'teacher_rel_path', 'stop_condition',
        'english_threshold', 'retain_arithmetic_threshold',
        'forget_arithmetic_threshold', 'arithmetic_train_file', 'eng_train_file'
    }
    train_params = {k: v for k, v in config.items() if k not in exclude}

    # Execute the core distillation tool
    partial_distill(
        eval_fn=eval_fn,
        stop_cond_fn=stop_wrapper,
        accelerator=accelerator,
        train_files=[config['eng_train_file'], config['arithmetic_train_file']],
        overwrite_ok=True,
        stopping_strategy='first_exhausted',
        **train_params
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Partial Distillation sweep via YAML config.")
    parser.add_argument("--setup", type=str, required=True, help="Setup ID from the YAML file.")
    args = parser.parse_args()

    # 1. Locate YAML file
    yaml_path = CONFIG_DIR / "arithmetic" / "distill.yaml"

    # 2. Load and expand configurations for the selected setup
    try:
        all_experiments = load_distill_configs(yaml_path, args.setup)
    except KeyError:
        print(f"❌ Error: Setup ID '{args.setup}' not found in 'setups' section of {yaml_path}")
        sys.exit(1)

    print(f"🚀 Initializing Partial Distillation Sweep for base setup: {args.setup}")
    print(f"Total experiments (Alpha x Beta x Seed): {len(all_experiments)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 3. Prepare task list for the parallel launcher
    task_list = [(eid, all_experiments) for eid in all_experiments.keys()]

    # 4. Launch in parallel across available GPUs
    parallel_launcher = get_parallel_launch_wrapper(launch_worker)
    launch_in_parallel_one_per_gpu(experiment_list=task_list, experiment_fn=parallel_launcher)