from localized_undo.tools.unlearn_langarith.graddiff import unlearn_graddiff
from localized_undo.tools.unlearn_langarith.maxent import unlearn_maxent
from localized_undo.tools.unlearn_langarith.rmu import unlearn_rmu
from accelerate import Accelerator
from localized_undo.utils.paths import WANDB_API_KEY_PATH, CONFIG_DIR
from localized_undo.utils.config_handler import load_unlearn_configs
from localized_undo.utils.validation_functions import get_arithmetic_eval_fn
from localized_undo.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

# Define which base methods we want to sweep
BASE_SETUPS_TO_RUN = ["gemma-2-0.3B_MaxEnt", "gemma-2-0.3B_RMU"]

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

    # Load WandB API key from local token file
    try:
        api_key = WANDB_API_KEY_PATH.read_text().strip()
    except Exception as e:
        if accelerator.is_main_process:
            print(f"[ERROR] Failed to read WandB key: {e}")
        return

    # Initialize the arithmetic evaluation function
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

    # Filter out keys used for path construction but not accepted by unlearn functions
    exclude = {
        'method',
        'wandb_api_key',
        'forget_train_file',
        'retain_train_file',
        'eng_valid_file',
        'model_name'
    }
    train_params = {k: v for k, v in config.items() if k not in exclude}

    # Select the unlearning function based on the method defined in YAML
    unlearn_fn = dispatch[config['method']]

    # Execute training loop with explicit file paths and unpacked hyperparameters
    unlearn_fn(
        model_name=config['model_name'],
        forget_train_file=config['forget_train_file'],
        retain_train_file=config['retain_train_file'],
        eval_fn=eval_fn,
        accelerator=accelerator,
        wandb_api_key=api_key,
        join_or_subsequence=True,
        **train_params
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