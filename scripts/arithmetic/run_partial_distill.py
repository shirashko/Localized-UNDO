import argparse
import sys
from datetime import datetime
from accelerate import Accelerator, init_empty_weights
import torch
from transformers import AutoModelForCausalLM

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

    # 4. Selection Logic
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

    # Re-authenticate in the worker process
    custom_login()

    # 1. Initialize evaluation function
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

    def stop_wrapper(student_eval_dict, teacher_eval_dict):
        return arithmetic_stop_cond_fn(student_eval_dict, teacher_eval_dict, config)

    # 2. Handle Localization Mask
    mask_path = config.get('noise_mask_path')
    noise_mask_tensor = None
    if mask_path:
        try:
            # Load mask to CPU initially
            raw_mask = torch.load(mask_path, map_location='cpu', weights_only=True)

            # Normalize mask keys by removing 'model.' prefix
            noise_mask_tensor = {k.replace("model.", ""): v for k, v in raw_mask.items()}

            # Alignment Check ---
            print(f"🔍 [Worker {exp_id}] Verifying mask alignment for {config['student_model_name']}...")

            with init_empty_weights():
                temp_model = AutoModelForCausalLM.from_pretrained(
                    config['student_model_name'],
                    cache_dir=config['cache_dir']
                )

            # Normalize model keys for comparison
            model_keys = {
                n.replace("module.", "").replace("student_model.", "").replace("model.", "")
                for n, p in temp_model.named_parameters() if p.requires_grad
            }
            mask_keys = set(noise_mask_tensor.keys())
            intersection = model_keys.intersection(mask_keys)

            print(f"[CHECK] Mask: {len(mask_keys)} keys | Model: {len(model_keys)} keys")
            print(f"[CHECK] Overlap: {len(intersection)} keys matched.")

            if len(intersection) == 0:
                example_model = sorted(list(model_keys))[0]
                example_mask = sorted(list(mask_keys))[0]
                print(f"⚠️ WARNING: Zero overlap! Model example: '{example_model}', Mask example: '{example_mask}'")

            del temp_model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Critical Error loading mask at {mask_path}: {e}")
            return

    # 3. Start Distillation
    partial_distill(
        teacher_model_name=config['teacher_model_name'],
        student_model_name=config['student_model_name'],
        train_files=[config['eng_train_file'], config['arithmetic_train_file']],
        interleave_probs=config['interleave_probs'],
        stopping_strategy=config.get('stopping_strategy', 'first_exhausted'),
        join_or_subsequence=config['join_or_subsequence'],
        eval_fn=eval_fn,
        stop_cond_fn=stop_wrapper,
        accelerator=accelerator,
        output_dir=config['output_dir'],
        cache_dir=config['cache_dir'],
        dataset_cache_dir=config['dataset_cache_dir'],
        seed=config['seed'],
        device=str(accelerator.device),
        batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        max_steps=config['max_steps'],
        num_warmup_steps=config['num_warmup_steps'],
        validation_steps=config['validation_steps'],
        save_checkpoint_steps=config['save_checkpoint_steps'],
        scheduler_type=config['scheduler_type'],
        min_lr=config['min_lr'],
        weight_decay=config['weight_decay'],
        gradient_clipping_threshold=config['gradient_clipping_threshold'],
        max_length=config['max_length'],
        use_wandb=config['use_wandb'],
        wandb_project=config['wandb_project'],
        wandb_run_name=config['wandb_run_name'],
        use_local_record=config['use_local_record'],
        path_local_record=config['path_local_record'],
        overwrite_ok=True,
        noise_alpha=config.get('noise_alpha', 0.0),
        noise_beta=config.get('noise_beta', 0.0),
        noise_type=config.get('noise_type', 'global'),
        shrink_perturb_repeat=config.get('shrink_perturb_repeat', False),
        noise_mask=noise_mask_tensor,
        noise_config=config.get('noise_mask_dir_name', None)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Partial Distillation sweep via YAML config.")
    parser.add_argument("--setup", type=str, required=True, help="Setup ID from the YAML file.")
    args = parser.parse_args()

    yaml_path = CONFIG_DIR / "arithmetic" / "partial_distill.yaml"

    try:
        all_experiments = load_distill_configs(yaml_path, args.setup)
    except KeyError:
        print(f"❌ Error: Setup ID '{args.setup}' not found in {yaml_path}")
        sys.exit(1)

    print(f"🚀 Initializing Partial Distillation Sweep: {args.setup}")
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    task_list = [(eid, all_experiments) for eid in all_experiments.keys()]
    parallel_launcher = get_parallel_launch_wrapper(launch_worker)
    launch_in_parallel_one_per_gpu(experiment_list=task_list, experiment_fn=parallel_launcher)