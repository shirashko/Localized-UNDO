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
ALL_EXP_CONFIGS = load_unlearn_configs(yaml_path, BASE_SETUPS_TO_RUN)


def launch_unlearning_run(setup_id):
    config = ALL_EXP_CONFIGS[setup_id]
    accelerator = Accelerator()

    # Load WandB API key
    api_key = WANDB_API_KEY_PATH.read_text().strip()

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

    # Dictionary mapping method names to their unlearn functions
    dispatch = {
        "GradDiff": unlearn_graddiff,
        "MaxEnt": unlearn_maxent,
        "RMU": unlearn_rmu
    }

    # Extract only relevant params for the function call
    exclude = {'method', 'wandb_api_key'}
    train_params = {k: v for k, v in config.items() if k not in exclude}

    # Execute the appropriate unlearn method
    unlearn_fn = dispatch[config['method']]
    unlearn_fn(
        eval_fn=eval_fn,
        accelerator=accelerator,
        wandb_api_key=api_key,
        join_or_subsequence=True,
        **train_params
    )


if __name__ == "__main__":
    experiments = [(sid,) for sid in ALL_EXP_CONFIGS.keys()]
    print(f"🚀 Launching {len(experiments)} unlearning experiments in parallel...")

    parallel_fn = get_parallel_launch_wrapper(launch_unlearning_run)
    launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)