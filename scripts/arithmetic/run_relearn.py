import argparse
from accelerate import Accelerator
from localized_undo.tools.relearn_langarith import relearn
from localized_undo.utils.paths import CONFIG_DIR
from localized_undo.utils.config_handler import load_relearn_configs
from localized_undo.utils.loss_functions import custom_login
from localized_undo.utils.validation_functions import get_arithmetic_eval_fn
from localized_undo.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

custom_login()

# --- Config for dynamic paths ---
seed = 111
beta = 0.1
alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
noise_mask = "global"  # or "delta-mask"
method = "MaxEnt"

# Build paths for all distilled variants
distilled_paths = [
    f"partial_distill_models_arith/gemma-2-0.1B_{method}-arithmetic-partial_distill-{noise_mask}-alpha_{a}-beta_{beta}-seed_{seed}"
    for a in alphas
]

MODELS_TO_RUN = [
                    'pretrained_models/gemma-2-0.1B_addition_subtraction+eng', # Pretrain Pure
                    'pretrained_models/gemma-2-0.1B_all_arithmetic+eng', # Pretrained Base
                    'unlearned_models/MaxEnt/gemma-2-0.1B_all_arithmetic+eng_lr_8.0e-05', # Unlearned MaxEnt
                ] + distilled_paths


def launch_relearn_worker(exp_id, all_configs):
    config = all_configs[exp_id]
    accelerator = Accelerator()
    custom_login()

    eval_fn = get_arithmetic_eval_fn(
        model_name=config['model_name'],
        eng_valid_file=config['eng_valid_file'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        num_wiki_batches=50,
        accelerator=accelerator
    )

    train_files = [config['first_train_file']]
    if config.get('second_train_file'):
        train_files.append(config['second_train_file'])

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
        save_checkpoint_steps=config.get('save_checkpoint_steps', 1500)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setups", nargs='+', default=["gemma-2-0.1B_train_only_forget"], help="Setup IDs")
    args = parser.parse_args()

    yaml_path = CONFIG_DIR / "arithmetic" / "relearn.yaml"
    all_experiments = load_relearn_configs(yaml_path, args.setups, MODELS_TO_RUN)

    print(f"🚀 Launching Relearning sweep for {len(all_experiments)} combinations...")

    task_list = [(eid, all_experiments) for eid in all_experiments.keys()]
    parallel_launcher = get_parallel_launch_wrapper(launch_relearn_worker)
    launch_in_parallel_one_per_gpu(experiment_list=task_list, experiment_fn=parallel_launcher)