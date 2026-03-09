import yaml
from localized_undo.utils.paths import MODEL_DIR, DATASET_DIR, CACHE_DIR


def load_relearn_configs(yaml_path, setup_ids, models_to_run):
    """
    Expands relearning setups across multiple models and multiple learning rates.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    relearn_lrs = data['relearn_lrs']
    expanded_experiments = {}

    for setup_id in setup_ids:
        base_cfg = data['default_config'].copy()
        base_cfg.update(data['setups'][setup_id])

        for model_rel_path in models_to_run:
            for lr in relearn_lrs:
                config = base_cfg.copy()
                lr_val = float(lr)
                config['learning_rate'] = lr_val
                config['min_lr'] = lr_val

                # Model Pathing
                # If it's a distilled model, it might not have the 'final_model' subfolder
                full_model_path = MODEL_DIR / model_rel_path
                if "distilled" not in str(full_model_path) and "pretrained" not in str(full_model_path):
                    if not str(full_model_path).endswith("final_model"):
                        full_model_path = full_model_path / "final_model"

                config['model_name'] = str(full_model_path)

                # Output Naming Logic
                safe_model_name = model_rel_path.replace('/', '_')
                exp_label = f"relearned_{safe_model_name}_{lr_val:.1e}"

                config['output_dir'] = str(MODEL_DIR / "relearned_models" / setup_id / exp_label)
                config['path_local_record'] = str(
                    MODEL_DIR / "local_records/relearned_models" / setup_id / f"{exp_label}.txt")
                config['wandb_run_name'] = f"{safe_model_name}_lr_{lr_val:.1e}"

                # Data Paths
                config['eng_valid_file'] = str(DATASET_DIR / "pretrain/valid_eng.jsonl")
                config['first_train_file'] = str(DATASET_DIR / config['first_train_file'])
                if config['second_train_file']:
                    config['second_train_file'] = str(DATASET_DIR / config['second_train_file'])

                config['cache_dir'] = str(CACHE_DIR)
                config['dataset_cache_dir'] = str(CACHE_DIR)

                unique_id = f"{setup_id}_{safe_model_name}_lr{lr_val}"
                expanded_experiments[unique_id] = config

    return expanded_experiments


def load_distill_configs(yaml_path, setup_id):
    """
    Expands a base setup into a list of configurations for alpha/beta/seed sweep.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    base_cfg = data['default_config'].copy()
    base_cfg.update(data['setups'][setup_id])
    base_cfg.update(data['stopping_criteria'])

    sweep = data['sweeps']
    expanded_experiments = {}

    for alpha in sweep['alphas']:
        for beta in sweep['betas']:
            # Handle seeds (if None, use default)
            seeds = sweep['seeds'] if sweep['seeds'] else [base_cfg['seed']]
            for seed in seeds:
                config = base_cfg.copy()
                config['noise_alpha'] = float(alpha)
                config['noise_beta'] = float(beta)
                config['seed'] = int(seed)

                # Dynamic Paths
                method = config['method']
                config['teacher_model_name'] = str(MODEL_DIR / config['teacher_rel_path'])
                config['student_model_name'] = config['teacher_model_name']  # Usually initialized from teacher

                # Path Naming
                path_suffix = f"-alpha_{alpha}-beta_{beta}-seed_{seed}"
                base_name = f"gemma-2-0.3B_{method}-arithmetic-partial_distill"

                config['output_dir'] = str(MODEL_DIR / "partial_distill_models_arith" / f"{base_name}{path_suffix}")
                config['path_local_record'] = str(
                    MODEL_DIR / "local_records/partial_distill_models_arith" / f"{base_name}{path_suffix}.txt")
                config['wandb_run_name'] = f"{setup_id}_alpha{alpha}_beta{beta}_seed{seed}"

                # Global Data Paths
                config['eng_train_file'] = str(DATASET_DIR / "pretrain/train_eng.jsonl")
                config['arithmetic_train_file'] = str(DATASET_DIR / "pretrain/train_all_arithmetic.jsonl")
                config['eng_valid_file'] = str(DATASET_DIR / "pretrain/valid_eng.jsonl")

                config['cache_dir'] = str(CACHE_DIR)
                config['dataset_cache_dir'] = str(CACHE_DIR)

                exp_id = f"{setup_id}_a{alpha}_b{beta}_s{seed}"
                expanded_experiments[exp_id] = config

    return expanded_experiments


def load_unlearn_configs(yaml_path, base_setup_ids):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    expanded_setups = {}
    for base_id in base_setup_ids:
        base_cfg = data['default_config'].copy()
        base_cfg.update(data['setups'][base_id])

        method = base_cfg['method']
        lr_range = data['lr_ranges'][method]

        for lr in lr_range:
            # FIX: Explicitly cast to float to handle string parsing from YAML
            lr_val = float(lr)
            setup_id = f"{base_id}_lr_{lr_val:.1e}"

            config = base_cfg.copy()
            config['learning_rate'] = lr_val
            config['min_lr'] = lr_val

            # Dynamic path construction (Ensuring consistency with your previous scripts)
            method_dir = config['method']
            config['model_name'] = str(MODEL_DIR / "pretrained_models/gemma-2-0.3B_all_arithmetic+eng/final_model")
            config['forget_train_file'] = str(DATASET_DIR / "pretrain/train_multiplication_division.jsonl")
            config['retain_train_file'] = str(DATASET_DIR / "pretrain/train_addition_subtraction.jsonl")
            config['eng_valid_file'] = str(DATASET_DIR / "pretrain/valid_eng.jsonl")

            config['output_dir'] = str(
                MODEL_DIR / f"unlearned_models/{method_dir}/gemma-2-0.3B_all_arithmetic+eng_lr_{lr_val:.1e}")
            config['path_local_record'] = str(
                MODEL_DIR / f"local_records/unlearned_models/{method_dir}/gemma-2-0.3B_all_arithmetic+eng_lr_{lr_val:.1e}.txt")

            config['cache_dir'] = str(CACHE_DIR)
            config['dataset_cache_dir'] = str(CACHE_DIR)
            config['wandb_run_name'] = f"lr_{lr_val:.1e}"

            expanded_setups[setup_id] = config

    return expanded_setups



def load_pretrain_config(yaml_path, setup_id):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    config = data['default_config'].copy()
    config.update(data['setups'][setup_id])

    m_id = config['model_id']
    a_type = config['arithmetic_type']

    config['model_name'] = str(MODEL_DIR / "random_init_models" / m_id)
    config['eng_train_file'] = str(DATASET_DIR / "pretrain" / "train_eng.jsonl")

    config['secondary_train_files'] = [str(DATASET_DIR / "pretrain" / f"train_{a_type}.jsonl")]
    config['eng_valid_file'] = str(DATASET_DIR / "pretrain" / "valid_eng.jsonl")

    config['output_dir'] = str(MODEL_DIR / "pretrained_models" / setup_id)
    config['path_local_record'] = str(MODEL_DIR / "local_records" / "pretrained_models" / f"{setup_id}.txt")

    config['cache_dir'] = str(CACHE_DIR)
    config['dataset_cache_dir'] = str(CACHE_DIR)
    config['wandb_project'] = setup_id

    return config