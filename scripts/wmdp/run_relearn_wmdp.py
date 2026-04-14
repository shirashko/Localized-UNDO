import os
from localized_undo.tools.relearn_wmdp import relearn
from localized_undo.utils.paths import CACHE_DIR, DATASET_DIR, WMDP_MODEL_DIR
from accelerate import Accelerator
from localized_undo.utils.loss_functions import custom_login
from localized_undo.utils.validation_functions import (
    get_wmdp_bio_eval_fn,
    get_wmdp_cyber_eval_fn,
    get_both_wmdp_eval_fn,
    get_loss_eval_fn,
)
from localized_undo.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

FINAL_RUN = True

SETUPS_TO_RUN = [
    "wmdp" # wmdp-shrink-perturb
]

eval_on_loss = False

MODELS_TO_RUN = {
    "gemma-2-2b": WMDP_MODEL_DIR / "gemma-2-2b",
}

SWEEP_SEEDS = [42] # [42, 43, 44, 45]

# DATA_TO_RUN maps experiment names to their data configurations.
# Structure: 
# 'experiment_name': (
#    [list_of_jsonl_files],      # Paths to dataset files
#    [interleave_probabilities], # Sampling weight for each file (must sum to 1.0)
#    [learning_rates_to_sweep]   # List of LRs for hyperparameter optimization
# )
DATA_TO_RUN = {
    # -------------------------------------------------------------------------
    # BASELINE: The 'bio-initial' run. 
    # Used as a control group to measure the model's state BEFORE relearning.
    # Note: LR is 0.0 and max_steps is set to 1 in launch_relearn() to ensure
    # we only perform evaluation on the original weights.
    # -------------------------------------------------------------------------
    # 'bio-initial' : ([
    #                             f'{DATASET_DIR}/pretrain/train_bio_remove_dataset.jsonl',
    #                             f'{DATASET_DIR}/pretrain/train_bio_retain_dataset.jsonl',
    #                         ], [.5, .5], [0e-5]),


    # -------------------------------------------------------------------------
    # MAIN EXPERIMENTS
    # -------------------------------------------------------------------------

    # forget/retain: Consists of a 50/50 mixed forget and retain WMDP corpora from the corresponding domain.

    # short
    'bio-forget/retain' : ([
                                f'{DATASET_DIR}/pretrain/train_bio_remove_dataset.jsonl',
                                f'{DATASET_DIR}/pretrain/train_bio_retain_dataset.jsonl',
                            ], [.5, .5], [1e-5, 2.15e-5, 4.62e-6]),

    # long
    # 'bio-forget/retain-long' : ([
    #                             f'{DATASET_DIR}/pretrain/train_bio_remove_dataset.jsonl',
    #                             f'{DATASET_DIR}/pretrain/train_bio_retain_dataset.jsonl',
    #                         ], [.5, .5], [1e-5, 2.15e-5, 4.62e-6]),


    # forget/retain-qa: Consists of a 50/50 mixed forget and retain question-answer dataset from the corresponding domain.

    # short
    # 'bio-forget/retain-qa': ([
    #                                 f'{DATASET_DIR}/pretrain/train_wmdp-bio_remove_dataset_qa.jsonl',
    #                                 f'{DATASET_DIR}/pretrain/train_wmdp-bio_retain_dataset_qa.jsonl'
    #                             ], [.5, .5], [1e-5, 2.15e-5, 4.62e-6]),

    # long
    # 'bio-forget/retain-qa-long': ([
    #                                 f'{DATASET_DIR}/pretrain/train_wmdp-bio_remove_dataset_qa.jsonl',
    #                                 f'{DATASET_DIR}/pretrain/train_wmdp-bio_retain_dataset_qa.jsonl'
    #                             ], [.5, .5], [4.62e-6, 2.15e-6]),

    # wiki-qa: Consists of a 50/25/25 mix of wikitext, forget question-answer, and retain question-answer from the corresponding domain.

    # short
    #  'bio-wiki-qa': ([
    #                             f'{DATASET_DIR}/pretrain/train_wikitext.jsonl',
    #                             f'{DATASET_DIR}/pretrain/train_wmdp-bio_remove_dataset_qa.jsonl',
    #                             f'{DATASET_DIR}/pretrain/train_wmdp-bio_retain_dataset_qa.jsonl',
    #                         ], [.5, .25, .25], [1e-5, 4.62e-6, 2.15e-6,]),
    # long
    # 'bio-wiki-qa-long': ([
    #                                 f'{DATASET_DIR}/pretrain/train_wikitext.jsonl',
    #                                 f'{DATASET_DIR}/pretrain/train_wmdp-bio_remove_dataset_qa.jsonl',
    #                                 f'{DATASET_DIR}/pretrain/train_wmdp-bio_retain_dataset_qa.jsonl',
    #                             ], [.5, .25, .25], [1e-5, 4.62e-6, 2.15e-6,]),
}

custom_login()

def validate_data_files(data_to_run):
    missing_files = []
    for experiment_name, (files, _probs, _lrs) in data_to_run.items():
        for file_path in files:
            if not os.path.isfile(file_path):
                missing_files.append((experiment_name, file_path))

    if missing_files:
        details = "\n".join(
            f"  - [{exp_name}] {path}" for exp_name, path in missing_files
        )
        raise FileNotFoundError(
            "Missing dataset files referenced in DATA_TO_RUN:\n" + details
        )

shared_setup = {
    'model_name': f"{WMDP_MODEL_DIR}/PATH",
    'train_files': [],
    'interleave_probs': [],
    'stopping_strategy': 'all_exhausted',
    'cache_dir': CACHE_DIR,
    'join_or_subsequence': True,
    'seed': 42,
    'device': "cuda",
    'batch_size': 20,
    'gradient_accumulation_steps': 1,
    'epochs': 5,
    'learning_rate': 'LR',
    'max_steps': 500,
    'num_warmup_steps': 1,
    'validation_steps': [10, 25, 50, 150, 500],
    'save_checkpoint_steps': 999,
    'scheduler_type': "cosine",
    'min_lr': 'LR',
    'weight_decay': 0.0,
    'gradient_clipping_threshold': 1.0,
    'max_length': 256,
    'use_wandb': True,
    'wandb_project': "relearn_CORPUS",
    'wandb_run_name': 'TBD',
    'use_local_record': True,
    'save_models': False,
}

setups = {
    "wmdp": {
        **shared_setup,
        'output_dir': f"{WMDP_MODEL_DIR}/relearned_models/TBD",
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/relearned_models/TBD.txt",
        'shrink_perturb_relearning': 0,
    },
    "wmdp-shrink-perturb": {
        **shared_setup,
        'output_dir': f"{WMDP_MODEL_DIR}/relearned_models/sp-TBD",
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/relearned_models/sp-TBD.txt",
        'shrink_perturb_relearning': .05,
    },
}

def launch_relearn(setup_id, lr, model, files, seed, eval_on_loss):
    name, path = model
    path = str(path).replace("SEED", str(seed))
    file_name, files_tup = files
    train_files, interleave_probs = files_tup
    if eval_on_loss and "qa" in file_name:
        name_tbd = 'loss_eval-qa_new/'
    elif eval_on_loss:
        name_tbd = 'loss_eval_new/'
    else:
        name_tbd = ''
    name_tbd += f"data-{file_name}_lr-{lr}_model-{name}_seed-{seed}_sparse-evals"

    current_setup = setups[setup_id].copy()

    if os.path.isabs(path):
        current_setup["model_name"] = path
    else:
        current_setup["model_name"] = current_setup["model_name"].replace("PATH", path)

    for key in ('path_local_record', 'output_dir', 'wandb_run_name'):
        current_setup[key] = current_setup[key].replace('TBD', name_tbd)

    current_setup['seed'] = seed
    if 'initial' in file_name:
        current_setup['validation_steps'] = [0]
        current_setup['max_steps'] = 1

    if 'long' in setup_id:
        assert current_setup['batch_size'] % 4 == 0
        current_setup['max_length'] *= 4
        current_setup['batch_size'] //= 4

    accelerator = Accelerator()
    train_percent = None
    if eval_on_loss:
        train_percent = .90
        assert 'retain' in train_files[1] and ('forget' in train_files[0] or 'remove' in train_files[0])
        eval_fn = get_loss_eval_fn(
            accelerator=accelerator,
        )
    elif 'bio' in file_name.lower():
        current_setup['wandb_project'] = current_setup['wandb_project'].replace('CORPUS', 'bio')
        eval_fn = get_wmdp_bio_eval_fn(accelerator, large_eval=FINAL_RUN, no_mmlu=not 'initial' in file_name)
    elif 'cyber' in file_name.lower():
        current_setup['wandb_project'] = current_setup['wandb_project'].replace('CORPUS', 'cyber')
        eval_fn = get_wmdp_cyber_eval_fn(accelerator, large_eval=FINAL_RUN, no_mmlu=not 'initial' in file_name)
    else:
        current_setup['wandb_project'] = current_setup['wandb_project'].replace('CORPUS', 'bio_baseline')
        eval_fn = get_both_wmdp_eval_fn(accelerator, large_eval=FINAL_RUN)

    relearn(
        model_name       = current_setup['model_name'],
        train_files      = train_files,
        interleave_probs = interleave_probs,
        stopping_strategy= current_setup['stopping_strategy'],
        output_dir       = current_setup['output_dir'],
        cache_dir        = current_setup['cache_dir'],
        dataset_cache_dir= current_setup['cache_dir'],
        eval_fn          = eval_fn,
        accelerator      = accelerator,
        join_or_subsequence   = current_setup['join_or_subsequence'],
        seed             = current_setup['seed'],
        device           = current_setup['device'],
        batch_size       = current_setup['batch_size'],
        gradient_accumulation_steps = current_setup['gradient_accumulation_steps'],
        epochs           = current_setup['epochs'],
        learning_rate    = lr,
        max_steps        = current_setup['max_steps'],   
        num_warmup_steps = current_setup['num_warmup_steps'],
        validation_steps = current_setup['validation_steps'],
        save_checkpoint_steps = current_setup['save_checkpoint_steps'],
        scheduler_type   = current_setup['scheduler_type'],  
        min_lr           = lr,          
        weight_decay     = current_setup['weight_decay'],    
        gradient_clipping_threshold = current_setup['gradient_clipping_threshold'], 
        max_length       = current_setup['max_length'],
        use_wandb        = current_setup['use_wandb'],
        wandb_project    = current_setup['wandb_project'],
        wandb_run_name   = current_setup['wandb_run_name'],
        use_local_record = current_setup['use_local_record'],
        path_local_record= current_setup['path_local_record'],
        overwrite_ok     = True,
        save_models      = current_setup['save_models'],
        shrink_perturb_relearning = current_setup['shrink_perturb_relearning'],
        train_percent = train_percent
    )

if __name__ == "__main__":
    # ----------------------------------------------------------------- #
    # Run all experiments, if possible in parallel
    # ----------------------------------------------------------------- #
    validate_data_files(DATA_TO_RUN)

    # Create list of the setups (arguments for run_experiment) for all the experiments we want to run 
    experiments = []
    for setup_id in SETUPS_TO_RUN:
        for seed in SWEEP_SEEDS:
            for model_name, model_path in MODELS_TO_RUN.items():
                for data_name, (files, probs, lrs) in DATA_TO_RUN.items():
                    for lr in lrs:
                        domain_in_model = "bio" in model_name.lower()
                        if not domain_in_model:
                            data_model_match = True
                        else:
                            data_model_match = "bio" in data_name and "bio" in model_name
                        if data_model_match:
                            experiments.append((setup_id, lr, (model_name, model_path), (data_name, (files, probs)), seed, eval_on_loss))
                        else:
                            print(f"Requested experiment {data_name} with model {model_name} does not match. Skipping.")

    # Gets a wrapper function compatable with the parallel launch function
    parallel_fn = get_parallel_launch_wrapper(launch_relearn)
    # calls run_experiment in parallel on a separate gpu for each experiment setup when a gpu is free
    launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)