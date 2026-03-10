from localized_undo.tools.unlearn_langarith.graddiff import unlearn_graddiff
from localized_undo.tools.unlearn_langarith.maxent import unlearn_maxent
from localized_undo.tools.unlearn_langarith.rmu import unlearn_rmu
from accelerate import Accelerator
from localized_undo.utils.paths import WANDB_API_KEY_PATH, CONFIG_DIR
from localized_undo.utils.config_handler import load_unlearn_configs
from localized_undo.utils.validation_functions import get_arithmetic_eval_fn
from localized_undo.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

# Define which base methods we want to sweep
BASE_SETUPS_TO_RUN = ["gemma-2-0.1B_MaxEnt"]

# Load expanded setups from YAML
yaml_path = CONFIG_DIR / "arithmetic" / "unlearn.yaml"
# This handler expands base setups into full learning rate variants
ALL_EXP_CONFIGS = load_unlearn_configs(yaml_path, BASE_SETUPS_TO_RUN)


def launch_unlearning_run(setup_id):
    """
    Worker function to run unlearning for a specific configuration.
    Designed to be called by the parallel launcher (one process per GPU).
    """
    config = ALL_EXP_CONFIGS[setup_id]
    accelerator = Accelerator()

    # Load WandB API key
    try:
        api_key = WANDB_API_KEY_PATH.read_text().strip()
    except Exception as e:
        if accelerator.is_main_process:
            print(f"[ERROR] Failed to read WandB key from {WANDB_API_KEY_PATH}: {e}")
        return

    # Initialize the arithmetic evaluation suite
    eval_fn = get_arithmetic_eval_fn(
        model_name=config['model_name'],
        eng_valid_file=config['eng_valid_file'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        cache_dir=config['cache_dir'],
        dataset_cache_dir=config['dataset_cache_dir'],
        num_wiki_batches=50,
        accelerator=accelerator
    )

    # Dictionary mapping method names to their respective tool functions
    dispatch = {
        "GradDiff": unlearn_graddiff,
        "MaxEnt": unlearn_maxent,
        "RMU": unlearn_rmu
    }

    unlearn_fn = dispatch[config['method']]

    # Explicitly mapping configuration keys to unlearn_maxent arguments.
    # This prevents the 'exclude' logic from accidentally filtering required research params.
    unlearn_fn(
        # Model & Paths
        model_name=config['model_name'],
        forget_train_file=config['forget_train_file'],
        retain_train_file=config['retain_train_file'],
        eval_fn=eval_fn,
        accelerator=accelerator,
        output_dir=config['output_dir'],
        cache_dir=config['cache_dir'],
        dataset_cache_dir=config['dataset_cache_dir'],

        # Data Processing
        join_or_subsequence=config.get('join_or_subsequence', True),
        use_retain=config.get('use_retain', True),
        max_length=config['max_length'],

        # Optimization Hyperparameters
        seed=config.get('seed', 42),
        device=config.get('device', 'cuda'),
        batch_size=config['batch_size'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 16),
        epochs=config.get('epochs', 1),
        learning_rate=config['learning_rate'],
        max_steps=config.get('max_steps', -1),
        num_warmup_steps=config.get('num_warmup_steps', 100),
        validation_steps=config.get('validation_steps', 50),
        save_checkpoint_steps=config.get('save_checkpoint_steps', 1500),
        scheduler_type=config.get('scheduler_type', "cosine"),
        min_lr=config.get('min_lr', 4e-5),
        weight_decay=config.get('weight_decay', 0.1),
        gradient_clipping_threshold=config.get('gradient_clipping_threshold', 1.0),

        # Logging & Analytics
        use_wandb=config.get('use_wandb', True),
        wandb_project=config['wandb_project'],
        wandb_run_name=config['wandb_run_name'],
        wandb_api_key=api_key,
        use_local_record=config.get('use_local_record', True),
        path_local_record=config['path_local_record'],

        # Research Specific Parameters (MaxEnt / SAM / RepNoise)
        balance_alpha=config.get('balance_alpha', 1.0),
        use_repnoise=config.get('use_repnoise', False),
        repnoise_beta=config.get('repnoise_beta', 0.001),
        repnoise_alpha=config.get('repnoise_alpha', 1.0),
        use_sam=config.get('use_sam', False),
        sam_rho=config.get('sam_rho', 0.05)
    )


if __name__ == "__main__":
    # Prepare list of experiment IDs for the parallel launcher
    experiments = [(sid,) for sid in ALL_EXP_CONFIGS.keys()]

    print(f"🚀 Launching {len(experiments)} unlearning experiments in parallel...")
    for sid in ALL_EXP_CONFIGS.keys():
        print(f"  - {sid} (LR: {ALL_EXP_CONFIGS[sid]['learning_rate']:.1e})")

    # Get a wrapper for the worker function and launch across available GPUs
    parallel_fn = get_parallel_launch_wrapper(launch_unlearning_run)
    launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)