import argparse
from accelerate import Accelerator
from localized_undo.tools.relearn_langarith import relearn
from localized_undo.utils.paths import CONFIG_DIR
from localized_undo.utils.config_handler import load_relearn_configs
from localized_undo.utils.loss_functions import custom_login
from localized_undo.utils.validation_functions import get_arithmetic_eval_fn
from localized_undo.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

# Global login
custom_login()

# The models you want to evaluate relearning for
MODELS_TO_RUN = [
    #'pretrained_models/gemma-2-0.3B_addition_subtraction+eng', # Pretrain Pure
    #'pretrained_models/gemma-2-0.3B_all_arithmetic+eng', # Pretrained Base
    #'unlearned_models/MaxEnt/gemma-2-0.3B_all_arithmetic+eng_lr_9.0e-05', # Unlearned MaxEnt

    #'distilled_models/pure/gemma-2-0.3B_all_arithmetic+eng', # Distilled Pure (Oracle)
    #'distilled_models/pure_from_base/gemma-2-0.3B_all_arithmetic+eng', # Distilled Impure (Oracle)
    #'distilled_models/base/gemma-2-0.3B_all_arithmetic+eng', # Distilled Base
    # 'distilled_models/MaxEnt/gemma-2-0.3B_all_arithmetic+eng', # Distilled MaxEnt

    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.1-beta_0.1-seed_123',
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.2-beta_0.1-seed_123',
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.3-beta_0.1-seed_123',
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.4-beta_0.1-seed_123',
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.5-beta_0.1-seed_123',
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.6-beta_0.1-seed_123',
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.7-beta_0.1-seed_123',
    #'partial_distill_models_arith/gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-alpha_0.8-beta_0.1-seed_123',
]


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

    # Filter for the core relearn tool
    exclude = {'first_train_file', 'second_train_file', 'eng_valid_file'}
    train_params = {k: v for k, v in config.items() if k not in exclude}

    relearn(
        train_files=train_files,
        eval_fn=eval_fn,
        accelerator=accelerator,
        **train_params
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setups", nargs='+', default=["gemma-2-0.3B_all_data"], help="List of setup IDs from YAML")
    args = parser.parse_args()

    yaml_path = CONFIG_DIR / "arithmetic" / "relearn.yaml"
    all_experiments = load_relearn_configs(yaml_path, args.setups, MODELS_TO_RUN)

    print(f"🚀 Launching Relearning sweep for {len(all_experiments)} combinations...")

    task_list = [(eid, all_experiments) for eid in all_experiments.keys()]
    parallel_launcher = get_parallel_launch_wrapper(launch_relearn_worker)
    launch_in_parallel_one_per_gpu(experiment_list=task_list, experiment_fn=parallel_launcher)