from accelerate import Accelerator
from pathlib import Path

from localized_undo.tools.partial_distill_wmdp import partial_distill
from localized_undo.utils.paths import CACHE_DIR, DATASET_DIR, WMDP_MODEL_DIR
from localized_undo.utils.loss_functions import custom_login
from localized_undo.utils.validation_functions import get_wmdp_bio_eval_fn
from localized_undo.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

SETUPS_TO_RUN = ["basic"]

MODELS = {
    'bio_rmu': f'{WMDP_MODEL_DIR}/unlearned_models/RMU/bio_lr_5.00e-05_alpha_0.50_seed_SEED/final_model',
    # 'bio_maxent': f'{WMDP_MODEL_DIR}/unlearned_models/MaxEnt/bio_lr_2.00e-05_alpha_0.30_seed_SEED/final_model',
    # 'base': f'{WMDP_MODEL_DIR}/gemma-2-2b'
}

SEEDS = [42]
SWEEP_LRS = [2e-5]
SWEEP_BASE_TEACH_PS = [0]
SWEEP_ALPHAS = [.25]

DATASET_BASE_DIR = DATASET_DIR / "pretrain"
FILES_NAMES = [
    "train_eng.jsonl",
    "train_wikipedia.jsonl",
    "train_magpie.jsonl",
    "train_magpie3-1.jsonl",
    "train_magpie-3.jsonl",
    "train_magpie-gemma2.jsonl",
    "train_magpie-phi3.jsonl",
    "train_magpie-qwen.jsonl",
    "train_magpie-qwen2.jsonl",
]
SWEEP_FILES = {
    'all data': ([DATASET_BASE_DIR / file for file in FILES_NAMES],
                 [.35, .36, .05, .04, .04, .04, .04, .04, .04]),
}


FINAL_RUN=True # Controls eval size and overwrite ok

custom_login()

def make_setup(**overrides):
    setup = {
        'teacher_model_name': 'TBD',
        'student_model_name': 'TBD',
        'train_files': [],
        'output_dir': f"{WMDP_MODEL_DIR}/distilled_partial_distill_models/TBD",
        'cache_dir': CACHE_DIR,
        'dataset_cache_dir': CACHE_DIR,
        'join_or_subsequence': True,
        'interleave_probs': [],
        'stopping_strategy': 'all_exhausted',
        'seed': 'TBD',
        'device': "cuda",
        'batch_size': 10,
        'gradient_accumulation_steps': 11,
        'epochs': 2,
        'learning_rate': 'lr',
        'max_steps': 3000,
        'num_warmup_steps': 50,
        'validation_steps': 200,
        'save_checkpoint_steps': 1000,
        'scheduler_type': "cosine",
        'min_lr': 'lr',
        'weight_decay': 0,
        'gradient_clipping_threshold': 1.0,
        'max_length': 1024,
        'noise_alpha': 'TBD',
        'noise_beta': 0,
        'use_wandb': True,
        'wandb_project': "wmdp-TBD-partial_distill",
        'wandb_run_name': None,
        'use_local_record': True,
        'path_local_record': f"{WMDP_MODEL_DIR}/local_records/partial_distill_models/TBD.txt",
        'shrink_perturb_repeat': False,
        'compile_mode': None,
        'layers_to_train': 'all',
        'layer_types_to_train': 'all',
        'base_teacher_name': None,
        'switch_teachers': False,
        'use_base_teacher_percent': 0,
        'use_activation_loss': False,
        'both_losses_act_loss_multiplier': None,
    }
    setup.update(overrides)
    return setup

setups = {
    "basic": make_setup(validation_steps=500), # batch size 44, 4 and gradient accumulation 10
    "activations-and-logits": make_setup(
        batch_size=5,
        gradient_accumulation_steps=22,
        use_activation_loss=True,
        both_losses_act_loss_multiplier=10,
    ),
    "activation-based": make_setup(
        use_activation_loss=True,
    ),
    "beta5": make_setup(
        output_dir=f"{WMDP_MODEL_DIR}/distilled_partial_distill_models/beta5-TBD",
        path_local_record=f"{WMDP_MODEL_DIR}/local_records/partial_distill_models/beta5-TBD.txt",
        noise_beta=0.5,
    ),
    "seq-1536": make_setup(
        output_dir=f"{WMDP_MODEL_DIR}/distilled_partial_distill_models/1536-TBD",
        path_local_record=f"{WMDP_MODEL_DIR}/local_records/partial_distill_models/1536-TBD.txt",
        batch_size=7,
        gradient_accumulation_steps=10,
        max_length=1536,
        noise_alpha=0.25,
    ),
    "seq-768": make_setup(
        output_dir=f"{WMDP_MODEL_DIR}/distilled_partial_distill_models/768-TBD",
        path_local_record=f"{WMDP_MODEL_DIR}/local_records/partial_distill_models/768-TBD.txt",
        batch_size=15,
        max_length=768,
        noise_alpha=0.25,
    ),
    "larger-batch": make_setup(
        output_dir=f"{WMDP_MODEL_DIR}/distilled_partial_distill_models/large-batch-TBD",
        path_local_record=f"{WMDP_MODEL_DIR}/local_records/partial_distill_models/large-batch-TBD.txt",
        gradient_accumulation_steps=22,
        noise_alpha=0.25,
        noise_beta=0.1,
    ),
    "switch-teachers": make_setup(
        output_dir=f"{WMDP_MODEL_DIR}/distilled_partial_distill_models/switch-teacher-TBD",
        path_local_record=f"{WMDP_MODEL_DIR}/local_records/partial_distill_models/switch-teacher-TBD.txt",
        noise_alpha=0.25,
        base_teacher_name='google/gemma-2-2b',
        switch_teachers=True,
        use_base_teacher_percent='TBD',
    ),
    "mixed-teachers": make_setup(
        output_dir=f"{WMDP_MODEL_DIR}/distilled_partial_distill_models/mixed-teachers-TBD",
        path_local_record=f"{WMDP_MODEL_DIR}/local_records/partial_distill_models/mixed-teachers-TBD.txt",
        batch_size=5,
        gradient_accumulation_steps=22,
        noise_alpha=0.25,
        base_teacher_name='google/gemma-2-2b',
        use_base_teacher_percent='TBD',
    ),
}

def wmdp_stop_cond_fn(student_eval_dict, teacher_eval_dict):
    def compute_avg_acc(eval_dict):
        values_list = []
        for key, value in eval_dict.items():
            if "mmlu" in key and "time" not in key:
                values_list.append(value)
        
        assert len(values_list) == 1
        return sum(values_list) / len(values_list)
    
    student_avg_acc = compute_avg_acc(student_eval_dict)
    teacher_eval_acc = compute_avg_acc(teacher_eval_dict)
    diff = teacher_eval_acc - student_avg_acc
    return diff < .03

## Defines the funcation that runs a single experiment
def run_experiment(setup_id, lr, train_files, model, seed, base_teacher_percent, alpha):
    data_name, files, interleave_probs = train_files
    model_name, model_path = model
    current_setup = setups[setup_id].copy()

    if isinstance(model_path, tuple):
        teacher_model_path = model_path[0].replace("SEED", str(seed))  # Use the first model path if it's a tuple
        student_model_path = model_path[1].replace("SEED", str(seed))  # Use the first model path if it's a tuple
        current_setup['teacher_model_name'] = teacher_model_path
        current_setup['student_model_name'] = student_model_path
    else:
        model_path = model_path.replace("SEED", str(seed))
        # Update paths and parameters with current alpha value
        current_setup['teacher_model_name'] = model_path
        current_setup['student_model_name'] = model_path
    
    accelerator = Accelerator()

    datatype = 'bio'
    eval_fn = get_wmdp_bio_eval_fn(accelerator, large_eval=FINAL_RUN)
   

    current_setup['learning_rate'] = lr
    current_setup['min_lr'] = lr / 10.
    current_setup['seed'] = seed

    model_name_for_path = model_name
    if datatype == "bio" and model_name_for_path.startswith("bio_"):
        model_name_for_path = model_name_for_path[len("bio_"):]
    tbd_path = f"{datatype}/{model_name_for_path}/{setup_id}-{data_name}-lr_{lr:2e}-base-p_{base_teacher_percent}-seed_{seed}-alpha{alpha}"

    current_setup['output_dir'] = current_setup['output_dir'].replace('TBD', tbd_path)
    current_setup['path_local_record'] = current_setup['path_local_record'].replace('TBD', tbd_path)
    partial_distill(
        teacher_model_name= current_setup['teacher_model_name'],
        student_model_name= current_setup['student_model_name'],
        train_files       = files,
        interleave_probs  = interleave_probs,
        stopping_strategy = current_setup['stopping_strategy'],
        join_or_subsequence=current_setup['join_or_subsequence'],
        eval_fn           = eval_fn,
        stop_cond_fn      = lambda student_eval_dict, teacher_eval_dict: False,
        accelerator       = accelerator,
        output_dir        = current_setup['output_dir'],
        cache_dir         = current_setup['cache_dir'],
        dataset_cache_dir = current_setup['dataset_cache_dir'],
        seed              = current_setup['seed'],
        device            = current_setup['device'],
        batch_size        = current_setup['batch_size'],
        gradient_accumulation_steps = current_setup['gradient_accumulation_steps'],
        epochs           = current_setup['epochs'],
        learning_rate    = current_setup['learning_rate'],
        max_steps        = current_setup['max_steps'],   
        num_warmup_steps = current_setup['num_warmup_steps'],
        validation_steps = current_setup['validation_steps'],
        save_checkpoint_steps = current_setup['save_checkpoint_steps'],
        scheduler_type   = current_setup['scheduler_type'],  
        min_lr           = current_setup['min_lr'],          
        weight_decay     = current_setup['weight_decay'],    
        gradient_clipping_threshold = current_setup['gradient_clipping_threshold'], 
        max_length       = current_setup['max_length'],
        use_wandb        = current_setup['use_wandb'],
        wandb_project    = current_setup['wandb_project'].replace('TBD', datatype),
        wandb_run_name   = tbd_path.replace('/', '_'),
        use_local_record = current_setup['use_local_record'],
        path_local_record= current_setup['path_local_record'],
        noise_alpha      = alpha,
        noise_beta       = current_setup['noise_beta'],
        shrink_perturb_repeat= current_setup['shrink_perturb_repeat'],
        overwrite_ok     = True,
        compile_mode     = current_setup['compile_mode'],
        layers_to_train = current_setup['layers_to_train'],
        layer_types_to_train=current_setup['layer_types_to_train'],

        base_teacher_name=current_setup['base_teacher_name'],
        switch_teachers=current_setup['switch_teachers'],
        use_base_teacher_percent=base_teacher_percent,
        use_activation_loss=current_setup['use_activation_loss'],
        both_losses_act_loss_multiplier=current_setup['both_losses_act_loss_multiplier']
    )

if __name__ == "__main__":
    
    # # ----------------------------------------------------------------- #
    # # Run all experiments, if possible in parallel
    # # ----------------------------------------------------------------- #
    # # Create list of the setups (arguments for run_experiment) for all the experiments we want to run 

    experiments = [(setup_id, lr, (name, val[0], val[1]), (model_name, model_path), seed, percent, alpha) for setup_id in SETUPS_TO_RUN for lr in SWEEP_LRS for name, val in SWEEP_FILES.items() for model_name, model_path in MODELS.items() for seed in SEEDS for percent in SWEEP_BASE_TEACH_PS for alpha in SWEEP_ALPHAS]
    # Gets a wrapper function compatable with the parallel launch function
    parallel_fn = get_parallel_launch_wrapper(run_experiment)
    # calls run_experiment in parallel on a separate gpu for each experiment setup when a gpu is free
    launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)
    