import argparse
import sys
from accelerate import Accelerator

from localized_undo.tools.relearn_wmdp import relearn
from localized_undo.utils.paths import CONFIG_DIR
from localized_undo.utils.config_handler import load_wmdp_relearn_configs
from localized_undo.utils.loss_functions import custom_login
from localized_undo.utils.validation_functions import (
    get_wmdp_cyber_eval_fn,
    get_wmdp_bio_eval_fn,
    get_loss_eval_fn
)
from localized_undo.utils.parallel_launch import (
    launch_in_parallel_one_per_gpu,
    get_parallel_launch_wrapper
)

# Global Research Flags
FINAL_RUN = True
custom_login()

# Model mapping with SEED placeholder for dynamic pathing
MODELS_TO_RUN = {
    'partial_distill_bio_RMU': 'distilled_partial_distill_models/general/bio_rmu/basic-all data-lr_1.000000e-05-seed_SEED/final_model',
    'partial_distill_cyber_RMU': 'distilled_partial_distill_models/general/cyber_rmu/basic-all data-lr_1.000000e-05-seed_SEED/final_model',
    'bio_RMU': 'saved_unlearned_models/RMU/bio_lr_5.00e-05_alpha_0.50_seed_SEED/final_model',
    'cyber_RMU': 'saved_unlearned_models/RMU/cyber_lr_2.00e-05_alpha_0.50_seed_SEED/final_model',
    'bio_SAM': 'unlearned_models/MaxEnt-SAM-kl/bio_lr_2.00e-05_alpha_0.30_seed_SEED/final_model',
    'cyber_SAM': 'unlearned_models/MaxEnt-SAM-kl/cyber_lr_2.00e-05_alpha_0.50_seed_SEED/final_model',
    'bio_repnoise': 'unlearned_models/MaxEnt-repnoise-kl/bio_lr_2.00e-05_alpha_0.30_seed_SEED/final_model',
    'cyber_repnoise': 'unlearned_models/MaxEnt-repnoise-kl/cyber_lr_2.00e-05_alpha_0.50_seed_SEED/final_model',
}


def launch_worker(exp_id, all_configs):
    """
    Worker function executed on a single GPU. 
    Handles evaluation function assignment and core execution with explicit arguments.
    """
    config = all_configs[exp_id]
    accelerator = Accelerator()

    # Required for multi-GPU process stability
    custom_login()

    # Determine Evaluation Function
    # If eval_on_loss is detected via the config (e.g. from the output_dir path)
    if 'loss_eval' in config['output_dir']:
        eval_fn = get_loss_eval_fn(accelerator=accelerator)
        config['train_percent'] = 0.90  # Set 90% train / 10% eval split
    else:
        # Determine domain from WandB project name or experiment ID
        is_initial = 'initial' in exp_id.lower()
        if 'cyber' in config['wandb_project'].lower():
            eval_fn = get_wmdp_cyber_eval_fn(
                accelerator, large_eval=FINAL_RUN, no_mmlu=not is_initial
            )
        else:
            eval_fn = get_wmdp_bio_eval_fn(
                accelerator, large_eval=FINAL_RUN, no_mmlu=not is_initial
            )

    relearn(
        model_name=config.get('model_name'),
        train_files=config.get('train_files'),
        eval_fn=eval_fn,  # Injected locally
        accelerator=accelerator,  # Injected locally
        join_or_subsequence=config.get('join_or_subsequence', True),
        interleave_probs=config.get('interleave_probs'),
        output_dir=config.get('output_dir'),
        cache_dir=config.get('cache_dir'),
        dataset_cache_dir=config.get('dataset_cache_dir'),
        seed=config.get('seed', 42),
        device=config.get('device', "cuda"),
        batch_size=config.get('batch_size', 4),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 16),
        epochs=config.get('epochs', 2),
        learning_rate=config.get('learning_rate', 4e-4),
        max_steps=config.get('max_steps', -1),
        num_warmup_steps=config.get('num_warmup_steps', 100),
        validation_steps=config.get('validation_steps', 50),
        save_checkpoint_steps=config.get('save_checkpoint_steps', 1500),
        scheduler_type=config.get('scheduler_type', "cosine"),
        min_lr=config.get('min_lr', 4e-5),
        weight_decay=config.get('weight_decay', 0.1),
        gradient_clipping_threshold=config.get('gradient_clipping_threshold', 1.0),
        max_length=config.get('max_length', 2048),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project'),
        wandb_run_name=config.get('wandb_run_name'),
        use_local_record=config.get('use_local_record', True),
        path_local_record=config.get('path_local_record'),
        stopping_strategy=config.get('stopping_strategy', 'first_exhausted'),
        overwrite_ok=True,  # Overriding with True for research sweeps
        save_models=config.get('save_models', True),
        shrink_perturb_relearning=config.get('shrink_perturb_relearning', False),
        train_percent=config.get('train_percent')
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel WMDP Relearning Runner.")
    parser.add_argument("--setups", nargs="+", required=True, help="YAML setups (e.g. wmdp wmdp-shrink-perturb).")
    parser.add_argument("--eval_on_loss", action="store_true", help="Enable loss-based evaluation.")
    args = parser.parse_args()

    yaml_path = CONFIG_DIR / "wmdp" / "relearn.yaml"

    # Generate Grid Sweep
    try:
        all_experiments = load_wmdp_relearn_configs(
            yaml_path,
            args.setups,
            MODELS_TO_RUN,
            eval_on_loss=args.eval_on_loss
        )
    except Exception as e:
        print(f"❌ Config Error: {e}")
        sys.exit(1)

    print(f"🚀 Initializing Relearning Sweep for: {args.setups}")
    print(f"Total experiment runs: {len(all_experiments)}")

    # Parallel Launch
    task_list = [(eid, all_experiments) for eid in all_experiments.keys()]
    parallel_launcher = get_parallel_launch_wrapper(launch_worker)

    launch_in_parallel_one_per_gpu(experiment_list=task_list, experiment_fn=parallel_launcher)