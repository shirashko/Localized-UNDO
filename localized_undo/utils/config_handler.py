import yaml
from localized_undo.utils.paths import MODEL_DIR, DATASET_DIR, CACHE_DIR, PROJECT_ROOT


def _initialize_base_config(data, setup_id):
    """
    Internal helper to merge defaults with setup-specific configs and
    standardize common paths and types.
    """
    if setup_id not in data['setups']:
        raise KeyError(f"Setup ID '{setup_id}' not found in the YAML configuration.")

    config = data['default_config'].copy()
    config.update(data['setups'][setup_id])

    # Standardize cache directories
    config['cache_dir'] = str(CACHE_DIR)
    config['dataset_cache_dir'] = str(CACHE_DIR)

    # Explicit casting for numerical hyperparameters
    for key in ['learning_rate', 'min_lr', 'weight_decay', 'noise_alpha', 'noise_beta']:
        if key in config and config[key] is not None:
            config[key] = float(config[key])

    return config


def load_relearn_configs(yaml_path, setup_ids, models_to_run):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    relearn_lrs = data['relearn_lrs']
    expanded_experiments = {}

    for setup_id in setup_ids:
        base_template = _initialize_base_config(data, setup_id)

        for model_rel_path in models_to_run:
            for lr in relearn_lrs:
                config = base_template.copy()
                lr_val = float(lr)
                config['learning_rate'] = lr_val
                config['min_lr'] = float(config.get('min_lr', lr_val))

                full_model_path = MODEL_DIR / model_rel_path
                if (full_model_path / "final_model").exists():
                    full_model_path = full_model_path / "final_model"

                if not full_model_path.exists():
                    raise FileNotFoundError(f"Base model for relearning not found: {full_model_path}")

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
                if config.get('second_train_file'):
                    config['second_train_file'] = str(DATASET_DIR / config['second_train_file'])

                unique_id = f"{setup_id}_{safe_model_name}_lr{lr_val}"
                expanded_experiments[unique_id] = config

    return expanded_experiments


def load_distill_configs(yaml_path, setup_id):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    base_template = _initialize_base_config(data, setup_id)
    if 'stopping_criteria' in data:
        base_template.update(data['stopping_criteria'])

    sweep = data['sweeps']
    expanded_experiments = {}

    for alpha in sweep['alphas']:
        for beta in sweep['betas']:
            seeds = sweep['seeds'] if sweep['seeds'] else [base_template['seed']]
            for seed in seeds:
                config = base_template.copy()
                config.update({
                    'noise_alpha': float(alpha),
                    'noise_beta': float(beta),
                    'seed': int(seed),
                    'noise_type': config.get('noise_type', 'global')
                })

                # Dynamic Paths
                method = config['method']
                teacher_path = MODEL_DIR / config['teacher_rel_path']
                if not teacher_path.exists():
                    raise FileNotFoundError(f"Teacher model not found: {teacher_path}")

                config['teacher_model_name'] = str(teacher_path)
                config['student_model_name'] = config['teacher_model_name']

                # Experiment Identification & Naming
                exp_id = f"{setup_id}_{config['noise_type']}_a{alpha}_b{beta}_s{seed}"
                path_suffix = f"-{config['noise_type']}-alpha_{alpha}-beta_{beta}-seed_{seed}"
                base_name = f"gemma-2-0.1B_{method}-arithmetic-partial_distill"

                config['output_dir'] = str(MODEL_DIR / "partial_distill_models_arith" / f"{base_name}{path_suffix}")
                config['path_local_record'] = str(
                    MODEL_DIR / "local_records/partial_distill_models_arith" / f"{base_name}{path_suffix}.txt")
                config['wandb_run_name'] = exp_id

                # Data Paths
                config['eng_train_file'] = str(DATASET_DIR / "pretrain/train_eng.jsonl")
                config['arithmetic_train_file'] = str(DATASET_DIR / "pretrain/train_all_arithmetic.jsonl")
                config['eng_valid_file'] = str(DATASET_DIR / "pretrain/valid_eng.jsonl")

                mask_file = config.get('noise_mask_file_name')
                if mask_file:
                    mask_path = PROJECT_ROOT / "localization_masks" / mask_file
                    if not mask_path.exists():
                        raise FileNotFoundError(f"Localization mask file missing: {mask_path}")
                    config['noise_mask_path'] = str(mask_path)

                expanded_experiments[exp_id] = config

    return expanded_experiments


def load_unlearn_configs(yaml_path, base_setup_ids):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    expanded_setups = {}
    for base_id in base_setup_ids:
        config_template = _initialize_base_config(data, base_id)
        method = config_template['method']
        lr_range = data['lr_ranges'][method]

        for lr in lr_range:
            lr_val = float(lr)
            setup_id = f"{base_id}_lr_{lr_val:.1e}"

            config = config_template.copy()
            config['learning_rate'] = lr_val
            config['min_lr'] = float(config.get('min_lr', lr_val))

            # Paths from YAML
            config['model_name'] = str(MODEL_DIR / config['model_rel_path'])
            config['forget_train_file'] = str(DATASET_DIR / config['forget_rel_path'])
            config['retain_train_file'] = str(DATASET_DIR / config['retain_rel_path'])
            config['eng_valid_file'] = str(DATASET_DIR / config['valid_rel_path'])

            # Output Directories
            model_slug = config['model_rel_path'].replace('/', '_')
            config['output_dir'] = str(MODEL_DIR / f"unlearned_models/{method}/{model_slug}_lr_{lr_val:.1e}")
            config['path_local_record'] = str(
                MODEL_DIR / f"local_records/unlearned_models/{method}/{model_slug}_lr_{lr_val:.1e}.txt")
            config['wandb_run_name'] = f"lr_{lr_val:.1e}"

            expanded_setups[setup_id] = config

    return expanded_setups


def load_pretrain_config(yaml_path, setup_id):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    config = _initialize_base_config(data, setup_id)

    m_id = config['model_id']
    a_type = config['arithmetic_type']

    config['model_name'] = str(MODEL_DIR / "random_init_models" / m_id)
    config['eng_train_file'] = str(DATASET_DIR / "pretrain" / "train_eng.jsonl")
    config['secondary_train_files'] = [str(DATASET_DIR / "pretrain" / f"train_{a_type}.jsonl")]
    config['eng_valid_file'] = str(DATASET_DIR / "pretrain" / "valid_eng.jsonl")

    config['output_dir'] = str(MODEL_DIR / "pretrained_models" / setup_id)
    config['path_local_record'] = str(MODEL_DIR / "local_records" / "pretrained_models" / f"{setup_id}.txt")

    return config