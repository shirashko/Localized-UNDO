import argparse
from pathlib import Path

from accelerate import Accelerator
from localized_undo.tools.relearn_langarith import relearn
from localized_undo.utils.paths import CONFIG_DIR, MODEL_DIR
from localized_undo.utils.config_handler import load_relearn_configs
from localized_undo.utils.loss_functions import custom_login
from localized_undo.utils.validation_functions import get_arithmetic_eval_fn
from localized_undo.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

# Authenticate with WandB and Hugging Face
custom_login()


def launch_relearn_worker(exp_id, all_configs):
    """
    Worker function to execute a single relearning experiment.
    Maps configuration from YAML to the explicit arguments of the relearn function.
    """
    config = all_configs[exp_id]
    accelerator = Accelerator()

    # Re-login in child process for safety
    custom_login()

    # Initialize the arithmetic evaluation function
    eval_fn = get_arithmetic_eval_fn(
        model_name=config['model_name'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        cache_dir=config['cache_dir'],
        dataset_cache_dir=config['dataset_cache_dir'],
        num_wiki_batches=50,
        eng_valid_file=config['eng_valid_file'],
        accelerator=accelerator
    )

    # Consolidate training files
    train_files = [config['first_train_file']]
    if config.get('second_train_file'):
        train_files.append(config['second_train_file'])

    # Prepare structured metadata for WandB tracking/grouping
    extra_config = {
        'base_model_version': config.get('base_model_version', 'unknown'),
        'parent_method': config.get('parent_method', 'baseline'),
        'parent_noise': config.get('parent_noise', 'none'),
        'parent_alpha': config.get('parent_alpha', 'none'),
        'learning_rate': config.get('learning_rate', 'unknown'),
    }

    # Execute the relearning process
    relearn(
        model_name=config['model_name'],
        train_files=train_files,
        eval_fn=eval_fn,
        accelerator=accelerator,
        output_dir=config['output_dir'],
        cache_dir=config['cache_dir'],
        dataset_cache_dir=config['dataset_cache_dir'],
        seed=config.get('seed', 42),
        batch_size=config['batch_size'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 16),
        epochs=config.get('epochs', 2),
        learning_rate=config['learning_rate'],
        max_steps=config.get('max_steps', -1),
        num_warmup_steps=config.get('num_warmup_steps', 100),
        validation_steps=config.get('validation_steps', 50),
        scheduler_type=config.get('scheduler_type', "cosine"),
        min_lr=config.get('min_lr', 4e-5),
        weight_decay=config.get('weight_decay', 0.1),
        gradient_clipping_threshold=config.get('gradient_clipping_threshold', 1.0),
        max_length=config['max_length'],
        join_or_subsequence=config.get('join_or_subsequence', True),
        interleave_probs=config.get('interleave_probs', [1.0]),
        stopping_strategy=config.get('stopping_strategy', 'first_exhausted'),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config['wandb_project'],
        wandb_run_name=config.get('wandb_run_name'),
        use_local_record=config.get('use_local_record', True),
        path_local_record=config['path_local_record'],
        overwrite_ok=config.get('overwrite_ok', False),
        save_models=config.get('save_models', True),
        save_checkpoint_steps=config.get('save_checkpoint_steps', 1500),
        extra_config=extra_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a parallelized relearning sweep.")
    parser.add_argument("--setups", nargs='+', default=["gemma-2-0.1B_train_only_forget"], help="List of setup IDs from the YAML file.")
    args = parser.parse_args()

    # 1. Load configuration and expand experiments
    yaml_path = CONFIG_DIR / "arithmetic" / "relearn.yaml"
    all_experiments = load_relearn_configs(yaml_path, args.setups)

    if not all_experiments:
        print("❌ No valid experiments were loaded. Check your YAML and setups.")
        exit(1)

    # 2. Safety Check: Verify that all referenced base models exist on disk
    missing_models = []
    # We use a set to avoid checking the same model path multiple times
    unique_model_paths = set(exp['model_name'] for exp in all_experiments.values())

    for model_path in unique_model_paths:
        p = Path(model_path)
        # Check if the path exists (load_relearn_configs already resolves final_model internally)
        if not p.exists():
            missing_models.append(model_path)

    if missing_models:
        print(f"❌ Error: The following base models are missing. Please check your unlearn/distill runs:")
        for m in missing_models:
            print(f"  - {m}")
        raise FileNotFoundError("One or more base models for relearning are missing.")

    # 3. Execution
    print(f"🚀 Launching Relearning sweep for {len(all_experiments)} combinations...")
    task_list = [(eid, all_experiments) for eid in all_experiments.keys()]
    parallel_launcher = get_parallel_launch_wrapper(launch_relearn_worker)
    launch_in_parallel_one_per_gpu(experiment_list=task_list, experiment_fn=parallel_launcher)