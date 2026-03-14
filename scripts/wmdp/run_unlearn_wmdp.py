import argparse
from accelerate import Accelerator

from localized_undo.tools.unlearn_wmdp.maxent import unlearn_maxent
from localized_undo.tools.unlearn_wmdp.rmu import unlearn_rmu
from localized_undo.tools.unlearn_wmdp.graddiff import unlearn_graddiff

from localized_undo.utils.paths import CONFIG_DIR
from localized_undo.utils.config_handler import load_wmdp_unlearn_configs
from localized_undo.utils.loss_functions import custom_login
from localized_undo.utils.validation_functions import get_wmdp_cyber_eval_fn, get_wmdp_bio_eval_fn
from localized_undo.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

FINAL_RUN = True
custom_login()


def launch_unlearning_worker(exp_id, all_configs):
    """
    Worker function executed on a specific GPU for a single config variant.
    """
    config = all_configs[exp_id]
    accelerator = Accelerator()

    # Select appropriate evaluation function based on domain
    if 'cyber' in exp_id.lower():
        eval_fn = get_wmdp_cyber_eval_fn(accelerator, large_eval=FINAL_RUN)
    else:
        eval_fn = get_wmdp_bio_eval_fn(accelerator, large_eval=FINAL_RUN)

    method = config['method']

    # Common arguments shared across unlearning implementations
    common_args = {
        'model_name': config['model_name'],
        'forget_train_file': config['forget_train_file'],
        'eval_fn': eval_fn,
        'accelerator': accelerator,
        'output_dir': config['output_dir'],
        'seed': config['seed'],
        'batch_size': config['batch_size'],
        'gradient_accumulation_steps': config['gradient_accumulation_steps'],
        'epochs': config['epochs'],
        'learning_rate': config['learning_rate'],
        'max_steps': config['max_steps'],
        'validation_steps': config['validation_steps'],
        'scheduler_type': config['scheduler_type'],
        'min_lr': config['min_lr'],
        'gradient_clipping_threshold': config['gradient_clipping_threshold'],
        'max_length': config['max_length'],
        'use_wandb': config['use_wandb'],
        'wandb_project': config['wandb_project'],
        'wandb_run_name': config['wandb_run_name'],
        'path_local_record': config['path_local_record'],
        'overwrite_ok': True
    }

    # Dispatch to the correct unlearning tool based on method
    if method in ["MaxEnt", "SAM", "repnoise"]:
        unlearn_maxent(
            **common_args,
            retain_files=config['retain_files'],
            interleave_probs=config.get('interleave_probs', [0.5, 0.5]),
            alpha=config['alpha'],
            use_sam=config.get('use_sam', False),
            use_repnoise=config.get('use_repnoise', False),
            stopping_strategy='first_exhausted',
            join_or_subsequence=True
        )
    elif method == "RMU":
        unlearn_rmu(
            **common_args,
            retain_files=config['retain_files'],
            interleave_probs=config.get('interleave_probs', [0.5, 0.5]),
            rmu_layers=config['rmu_layers'],
            end_layer=config.get('end_layer', 15),
            alpha=config['alpha'],
            c=config['c'],
            ga_gd=config.get('ga_gd', True),
            stopping_strategy='first_exhausted',
            join_or_subsequence=True
        )
    elif method == "GradDiff":
        unlearn_graddiff(
            **common_args,
            retain_train_file=config['retain_files'][0],  # Usually takes single file
            alpha=config['alpha'],
            ga_gd=config.get('ga_gd', True),
            join_or_subsequence=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel WMDP Unlearning Runner.")
    parser.add_argument("--setups", nargs="+", required=True, help="List of setup IDs from YAML.")
    args = parser.parse_args()

    yaml_path = CONFIG_DIR / "wmdp" / "unlearn.yaml"
    all_experiments = load_wmdp_unlearn_configs(yaml_path, args.setups)

    # Create task list (exp_id, total_configs_dict) for the parallel launcher
    task_list = [(eid, all_experiments) for eid in all_experiments.keys()]

    # Launch experiments in parallel across available GPUs
    parallel_launcher = get_parallel_launch_wrapper(launch_unlearning_worker)
    launch_in_parallel_one_per_gpu(experiment_list=task_list, experiment_fn=parallel_launcher)