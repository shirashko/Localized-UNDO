import argparse
import sys
from accelerate import Accelerator

from localized_undo.tools.partial_distill_wmdp import partial_distill
from localized_undo.utils.paths import CONFIG_DIR, WMDP_MODEL_DIR
from localized_undo.utils.config_handler import load_wmdp_distill_configs
from localized_undo.utils.loss_functions import custom_login
from localized_undo.utils.validation_functions import (
    get_wmdp_cyber_eval_fn,
    get_wmdp_bio_eval_fn,
    get_both_wmdp_eval_fn
)
from localized_undo.utils.parallel_launch import (
    launch_in_parallel_one_per_gpu,
    get_parallel_launch_wrapper
)

# Set global flags for research scale
FINAL_RUN = True
custom_login()

# Centralized model mapping for teacher/student initialization
MODELS = {
    'bio_rmu': f'{WMDP_MODEL_DIR}/saved_unlearned_models/RMU/bio_lr_5.00e-05_alpha_0.50_seed_SEED/final_model',
    'bio_maxent': f'{WMDP_MODEL_DIR}/saved_unlearned_models/MaxEnt/bio_lr_2.00e-05_alpha_0.30_seed_SEED/final_model',
    'cyber_rmu': f'{WMDP_MODEL_DIR}/saved_unlearned_models/RMU/cyber_lr_2.00e-05_alpha_0.50_seed_SEED/final_model',
    'cyber_maxent': f'{WMDP_MODEL_DIR}/saved_unlearned_models/MaxEnt/cyber_lr_2.00e-05_alpha_0.20_seed_SEED/final_model',
}

def launch_worker(exp_id, all_configs):
    """
    Worker process target. Handles single GPU experiment execution with explicit arguments.
    """
    config = all_configs[exp_id]
    accelerator = Accelerator()

    # Re-authenticate for distributed environment stability
    custom_login()

    # Domain-specific evaluation assignment
    domain = config.get('domain', 'both')
    if domain == 'cyber':
        eval_fn = get_wmdp_cyber_eval_fn(accelerator, large_eval=FINAL_RUN)
    elif domain == 'bio':
        eval_fn = get_wmdp_bio_eval_fn(accelerator, large_eval=FINAL_RUN)
    else:
        eval_fn = get_both_wmdp_eval_fn(accelerator, large_eval=FINAL_RUN)

    partial_distill(
        teacher_model_name=config.get('teacher_model_name'),
        student_model_name=config.get('student_model_name'),
        train_files=config.get('train_files'),
        interleave_probs=config.get('interleave_probs'),
        stopping_strategy=config.get('stopping_strategy'),
        join_or_subsequence=config.get('join_or_subsequence'),
        eval_fn=eval_fn,
        stop_cond_fn=lambda s, t: False,
        accelerator=accelerator,
        output_dir=config.get('output_dir'),
        cache_dir=config.get('cache_dir'),
        dataset_cache_dir=config.get('dataset_cache_dir'),
        seed=config.get('seed'),
        device=config.get('device'),
        batch_size=config.get('batch_size'),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps'),
        epochs=config.get('epochs'),
        learning_rate=config.get('learning_rate'),
        max_steps=config.get('max_steps'),
        num_warmup_steps=config.get('num_warmup_steps'),
        validation_steps=config.get('validation_steps'),
        save_checkpoint_steps=config.get('save_checkpoint_steps'),
        scheduler_type=config.get('scheduler_type'),
        min_lr=config.get('min_lr'),
        weight_decay=config.get('weight_decay'),
        gradient_clipping_threshold=config.get('gradient_clipping_threshold'),
        max_length=config.get('max_length'),
        use_wandb=config.get('use_wandb'),
        wandb_project=config.get('wandb_project'),
        wandb_run_name=config.get('wandb_run_name'),
        use_local_record=config.get('use_local_record'),
        path_local_record=config.get('path_local_record'),
        overwrite_ok=True, # Explicitly True for research sweeps
        noise_alpha=config.get('noise_alpha', 0.0),
        noise_beta=config.get('noise_beta', 0.0),
        shrink_perturb_repeat=config.get('shrink_perturb_repeat', False),
        compile_mode=config.get('compile_mode'),
        layers_to_train=config.get('layers_to_train', ['all']),
        layer_types_to_train=config.get('layer_types_to_train', ['all']),
        base_teacher_name=config.get('base_teacher_name'),
        switch_teachers=config.get('switch_teachers', False),
        use_base_teacher_percent=config.get('use_base_teacher_percent', 0),
        use_activation_loss=config.get('use_activation_loss', False),
        both_losses_act_loss_multiplier=config.get('both_losses_act_loss_multiplier')
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel WMDP Partial Distillation Sweep.")
    parser.add_argument("--setups", nargs="+", required=True, help="Setup IDs from YAML (e.g., basic beta5).")
    args = parser.parse_args()

    # 1. Load YAML and expand into full grid search experiments
    yaml_path = CONFIG_DIR / "wmdp" / "partial_distill.yaml"
    try:
        all_experiments = load_wmdp_distill_configs(yaml_path, args.setups, MODELS)
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)

    print(f"🚀 Initializing Sweep for setups: {args.setups}")
    print(f"Total experiments to run: {len(all_experiments)}")

    # 2. Map experiments to tasks for GPU scheduling
    task_list = [(eid, all_experiments) for eid in all_experiments.keys()]

    # 3. Parallel Execution across available hardware
    parallel_launcher = get_parallel_launch_wrapper(launch_worker)
    launch_in_parallel_one_per_gpu(experiment_list=task_list, experiment_fn=parallel_launcher)