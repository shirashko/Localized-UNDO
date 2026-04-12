from accelerate import Accelerator

from localized_undo.tools.unlearn_wmdp.graddiff import unlearn_graddiff
from localized_undo.tools.unlearn_wmdp.maxent import unlearn_maxent
from localized_undo.tools.unlearn_wmdp.rmu import unlearn_rmu
from localized_undo.utils.paths import CACHE_DIR, DATASET_DIR, WMDP_MODEL_DIR
from localized_undo.utils.loss_functions import custom_login
from localized_undo.utils.validation_functions import get_wmdp_bio_eval_fn
from localized_undo.utils.parallel_launch import launch_in_parallel_one_per_gpu, get_parallel_launch_wrapper

FINAL_RUN = True  # Controls eval size and overwrite ok

LR_RANGES = {
    "bio_MaxEnt": [2e-5],  # [5e-5],
    "bio_RMU": [5e-5],  # [1e-4]
    "bio_repnoise": [2e-5],
    "bio_SAM": [2e-5],
}
ALPHA_RANGES = {
    "bio_MaxEnt": [0.3, 0.1],  # [.75],
    "bio_RMU": [0.5],  # [.1, .3]
    "bio_repnoise": [0.3, 0.1],
    "bio_SAM": [0.3, 0.1],
}

SEEDS = [42, 43, 44, 45]

BASE_SETUPS = [
    "bio_RMU",
    "bio_MaxEnt",
    "bio_repnoise",
    "bio_SAM",
]
custom_login()

MODEL = f"{WMDP_MODEL_DIR}/gemma-2-2b"

DATASET_DIR = f"{DATASET_DIR}/pretrain"
BIO_FORGET = f"{DATASET_DIR}/train_wmdp-bio_remove_dataset_qa.jsonl"
BIO_RETAIN = [
    f"{DATASET_DIR}/train_wmdp-bio_retain_dataset_qa.jsonl",
    f"{DATASET_DIR}/train_wikitext.jsonl",
]
BIO_RMU_RETAIN = [
    f"{DATASET_DIR}/train_wikitext.jsonl",
    f"{DATASET_DIR}/train_wmdp-bio_retain_dataset_qa.jsonl",
]

shared_base_setups = {
    "model_name": MODEL,
    "forget_train_file": BIO_FORGET,
    "interleave_probs": [0.5, 0.5],
    "cache_dir": CACHE_DIR,
    "dataset_cache_dir": CACHE_DIR,
    "seed": 42,
    "device": "cuda",
    "epochs": 5,
    "learning_rate": "TBD",
    "min_lr": "TBD",
    "max_steps": 90,
    "num_warmup_steps": 0,
    "validation_steps": 100,
    "save_checkpoint_steps": -1,
    "scheduler_type": "cosine",
    "weight_decay": 0.0,
    "gradient_clipping_threshold": 1.0,
    "max_length": 256,
    "use_wandb": True,
    "wandb_run_name": None,
    "use_local_record": True,
}

shared_maxent_base_setpup = {
    **shared_base_setups,
    "retain_files": BIO_RETAIN,
    "use_retain": True,
    "use_retain_kl": True,
}

base_setups = {
    "bio_RMU": {
        **shared_base_setups,
        "retain_files": BIO_RMU_RETAIN,
        "stopping_strategy": "first_exhausted",
        "output_dir": f"{WMDP_MODEL_DIR}/unlearned_models/RMU/bio_TBD",
        "path_local_record": f"{WMDP_MODEL_DIR}/local_records/unlearned_models/RMU/bio_TBD.txt",
        "wandb_project": "bio_unlearn_RMU",
        "batch_size": 8,
        "gradient_accumulation_steps": 5,
        "ga_gd": True,
        "rmu_layers": [10, 11, 12, 13, 14, 15],
        "end_layer": 15,
        "c": 80,
    },
    "bio_MaxEnt": {
        **shared_maxent_base_setpup,
        "output_dir": f"{WMDP_MODEL_DIR}/unlearned_models/MaxEnt/bio_TBD",
        "path_local_record": f"{WMDP_MODEL_DIR}/local_records/unlearned_models/MaxEnt/bio_TBD.txt",
        "wandb_project": "bio_unlearn_MaxEnt",
        "batch_size": 4,
        "gradient_accumulation_steps": 10,
        "use_repnoise": False,
        "use_sam": False,
    },
    "bio_repnoise": {
        **shared_maxent_base_setpup,
        "output_dir": f"{WMDP_MODEL_DIR}/unlearned_models/MaxEnt-repnoise/bio_TBD",
        "path_local_record": f"{WMDP_MODEL_DIR}/local_records/unlearned_models/MaxEnt-repnoise/bio_TBD.txt",
        "wandb_project": "bio_unlearn_MaxEnt-repnoise",
        "batch_size": 1,
        "gradient_accumulation_steps": 40,
        "use_repnoise": True,
        "use_sam": False,
    },
    "bio_SAM": {
        **shared_maxent_base_setpup,
        "output_dir": f"{WMDP_MODEL_DIR}/unlearned_models/MaxEnt-SAM-kl/bio_TBD",
        "path_local_record": f"{WMDP_MODEL_DIR}/local_records/unlearned_models/MaxEnt-SAM-kl/bio_TBD.txt",
        "wandb_project": "bio_unlearn_MaxEnt-SAM",
        "batch_size": 4,
        "gradient_accumulation_steps": 10,
        "use_repnoise": False,
        "use_sam": True,
    },
}


def create_lr_alpha_variant(base_setup_id, learning_rate, alpha, seed):
    tbd_str = f"lr_{learning_rate:.2e}_alpha_{alpha:.2f}_seed_{seed}"
    new_setup_id = f"{base_setup_id}_{tbd_str}"
    setup_config = base_setups[base_setup_id].copy()
    setup_config["learning_rate"] = learning_rate
    setup_config["min_lr"] = learning_rate
    setup_config["alpha"] = alpha
    setup_config["seed"] = seed
    setup_config["output_dir"] = setup_config["output_dir"].replace("TBD", tbd_str)
    setup_config["path_local_record"] = setup_config["path_local_record"].replace("TBD", tbd_str)
    setup_config["wandb_run_name"] = tbd_str
    return new_setup_id, setup_config


setups = {}
SETUPS_TO_RUN = []
for base_setup_id in BASE_SETUPS:
    lr_range = LR_RANGES[base_setup_id]
    alpha_range = ALPHA_RANGES[base_setup_id]
    for lr in lr_range:
        for alpha in alpha_range:
            for seed in SEEDS:
                new_setup_id, setup_config = create_lr_alpha_variant(base_setup_id, lr, alpha, seed)
                setups[new_setup_id] = setup_config
                SETUPS_TO_RUN.append(new_setup_id)


def _unlearn_tail(cfg):
    """Training / IO fields passed unchanged into unlearn_graddiff, unlearn_maxent, unlearn_rmu."""
    keys = (
        "model_name",
        "output_dir",
        "cache_dir",
        "dataset_cache_dir",
        "seed",
        "device",
        "batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "learning_rate",
        "max_steps",
        "num_warmup_steps",
        "validation_steps",
        "save_checkpoint_steps",
        "scheduler_type",
        "min_lr",
        "weight_decay",
        "gradient_clipping_threshold",
        "max_length",
        "use_wandb",
        "wandb_project",
        "wandb_run_name",
        "use_local_record",
        "path_local_record",
    )
    return {k: cfg[k] for k in keys}


def launch_unlearning_run(setup_id):
    cfg = setups[setup_id]
    accelerator = Accelerator()
    eval_fn = get_wmdp_bio_eval_fn(accelerator, large_eval=FINAL_RUN)

    common = {
        **_unlearn_tail(cfg),
        "eval_fn": eval_fn,
        "accelerator": accelerator,
        "join_or_subsequence": True,
    }

    if "_GradDiff" in setup_id:
        unlearn_graddiff(
            **common,
            forget_train_file=cfg["forget_train_file"],
            retain_train_file=cfg["retain_train_file"],
            ga_gd=cfg["ga_gd"],
            alpha=cfg["alpha"],
            overwrite_ok=not FINAL_RUN,
        )
    elif "_MaxEnt" in setup_id or "_SAM" in setup_id or "_repnoise" in setup_id:
        print(f"Running MaxEnt with learning rate {cfg['learning_rate']}")
        unlearn_maxent(
            **common,
            forget_train_file=cfg["forget_train_file"],
            retain_files=cfg["retain_files"],
            interleave_probs=cfg["interleave_probs"],
            stopping_strategy="first_exhausted",
            use_retain=cfg["use_retain"],
            use_retain_kl=cfg["use_retain_kl"],
            alpha=cfg["alpha"],
            overwrite_ok=True,
            use_sam=cfg["use_sam"],
            use_repnoise=cfg["use_repnoise"],
        )
    elif "_RMU" in setup_id:
        unlearn_rmu(
            **common,
            forget_train_file=cfg["forget_train_file"],
            retain_files=cfg["retain_files"],
            interleave_probs=cfg["interleave_probs"],
            stopping_strategy=cfg["stopping_strategy"],
            ga_gd=cfg["ga_gd"],
            rmu_layers=cfg["rmu_layers"],
            end_layer=cfg["end_layer"],
            alpha=cfg["alpha"],
            c=cfg["c"],
            overwrite_ok=True,
        )


if __name__ == "__main__":
    print(f"Running {len(SETUPS_TO_RUN)} experiments with learning rate search:")

    for sid in SETUPS_TO_RUN:
        print(f"  - {sid} (LR: {setups[sid]['learning_rate']:.1e})")

    # Create list of the setups (arguments for run_experiment) for all the experiments we want to run
    experiments = [(sid,) for sid in SETUPS_TO_RUN]

    # Gets a wrapper function compatable with the parallel launch function
    parallel_fn = get_parallel_launch_wrapper(launch_unlearning_run)

    # calls run_experiment in parallel on a separate gpu for each experiment setup when a gpu is available
    launch_in_parallel_one_per_gpu(experiment_list=experiments, experiment_fn=parallel_fn)
