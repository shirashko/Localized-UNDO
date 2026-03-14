import yaml
from localized_undo.utils.paths import MODEL_DIR, DATASET_DIR, CACHE_DIR, PROJECT_ROOT, WMDP_MODEL_DIR


def _initialize_base_config(data, setup_id):
    """
    Unified inheritance logic: Starts with default_config and
    overwrites with setup-specific values if they exist.
    """
    if 'setups' not in data or setup_id not in data['setups']:
        raise KeyError(f"Setup ID '{setup_id}' not found in the YAML configuration.")

    # Start with a deep copy of the defaults
    config = data['default_config'].copy()

    # Retrieve overrides. Handle the case where the setup is empty/None in YAML
    setup_overrides = data['setups'].get(setup_id)
    if setup_overrides:
        config.update(setup_overrides)

    # Standardize cache directories to the system paths
    config['cache_dir'] = str(CACHE_DIR)
    config['dataset_cache_dir'] = str(CACHE_DIR)

    float_keys = [
        'learning_rate', 'min_lr', 'noise_alpha', 'noise_beta',
        'weight_decay', 'gradient_clipping_threshold',
        'both_losses_act_loss_multiplier', 'use_base_teacher_percent'
    ]
    for key in float_keys:
        if key in config and config[key] is not None:
            config[key] = float(config[key])

    int_keys = ['batch_size', 'gradient_accumulation_steps', 'max_steps', 'seed', 'max_length', 'epochs']
    for key in int_keys:
        if key in config and config[key] is not None:
            config[key] = int(config[key])

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


def load_wmdp_unlearn_configs(yaml_path, setup_ids):
    """
    Expands WMDP unlearning setups into multiple experiments based on
    learning rate ranges, alpha ranges, and seeds defined in YAML.
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Retrieve seeds from sweep section or default config
    seeds = data.get('sweeps', {}).get('seeds', [data.get('default_config', {}).get('seed', 42)])
    expanded_experiments = {}

    for base_id in setup_ids:
        # Merges default_config with specific setup config
        config_template = _initialize_base_config(data, base_id)

        method = config_template['method']
        # Detect domain (cyber/bio) from the setup ID string
        domain = 'cyber' if 'cyber' in base_id.lower() else 'bio'

        # Fetch LR and Alpha ranges based on method and domain
        lrs = data.get('lr_ranges', {}).get(method, [2e-05])
        alpha_key = f"{method}_{domain}"
        alphas = data.get('alpha_ranges', {}).get(alpha_key, [0.5])

        for lr in lrs:
            for alpha in alphas:
                for seed in seeds:
                    config = config_template.copy()
                    lr_val = float(lr)
                    alpha_val = float(alpha)

                    # Update core hyperparameters for this iteration
                    config.update({
                        'learning_rate': lr_val,
                        'min_lr': lr_val,
                        'alpha': alpha_val,
                        'seed': int(seed)
                    })

                    # Unique identifier for directory naming and tracking
                    params_str = f'lr_{lr_val:.2e}_alpha_{alpha_val:.2f}_seed_{seed}'

                    # Construct full model path from relative path in YAML
                    if 'model_rel_path' in config:
                        config['model_name'] = str(WMDP_MODEL_DIR / config['model_rel_path'])

                    # Set dynamic output directories and record paths
                    config['output_dir'] = str(MODEL_DIR / f"unlearned_models/{method}/{domain}_{params_str}")
                    config['path_local_record'] = str(
                        MODEL_DIR / f"local_records/unlearned_models/{method}/{domain}_{params_str}.txt")

                    # Set informative WandB run name
                    config['wandb_run_name'] = f"{method}_{domain}_{params_str}"

                    # Construct full data paths
                    config['forget_train_file'] = str(DATASET_DIR / config['forget_rel_path'])
                    if 'retain_files' in config:
                        config['retain_files'] = [str(DATASET_DIR / p) for p in config['retain_files']]

                    unique_id = f"{base_id}_{params_str}"
                    expanded_experiments[unique_id] = config

    return expanded_experiments


def load_wmdp_distill_configs(yaml_path, setup_ids, model_map):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    sweep = data.get('sweeps', {})
    dataset_info = data.get('datasets', {})
    expanded_experiments = {}

    for setup_id in setup_ids:
        config_template = _initialize_base_config(data, setup_id)

        for model_name, model_path_template in model_map.items():
            for dataset_name, d_val in dataset_info.items():
                for seed in sweep.get('seeds', [42]):
                    for lr in sweep.get('lrs', [2e-5]):
                        for alpha in sweep.get('alphas', [0.25]):
                            for teacher_pct in sweep.get('base_teacher_percents', [0]):

                                config = config_template.copy()

                                if isinstance(model_path_template, tuple):
                                    config['teacher_model_name'] = model_path_template[0].replace("SEED", str(seed))
                                    config['student_model_name'] = model_path_template[1].replace("SEED", str(seed))
                                else:
                                    path = model_path_template.replace("SEED", str(seed))
                                    config['teacher_model_name'] = path
                                    config['student_model_name'] = path

                                config.update({
                                    'learning_rate': float(lr),
                                    'min_lr': float(lr / 10.0),
                                    'seed': int(seed),
                                    'noise_alpha': float(alpha),
                                    'use_base_teacher_percent': float(teacher_pct)
                                })

                                # Dataset logic
                                config['train_files'] = [str(DATASET_DIR / f) for f in d_val['files']]
                                config['interleave_probs'] = d_val['probs']
                                config['domain'] = 'cyber' if 'cyber' in model_name else (
                                    'bio' if 'bio' in model_name else 'both')

                                # Path Naming
                                exp_slug = f"{config['domain']}/{model_name}/{setup_id}-{dataset_name}-lr_{lr:2e}-seed_{seed}-tp_{teacher_pct}"
                                config['output_dir'] = str(WMDP_MODEL_DIR / "distilled_models" / exp_slug)
                                config['path_local_record'] = str(WMDP_MODEL_DIR / "local_records" / f"{exp_slug}.txt")
                                config['wandb_run_name'] = exp_slug.replace('/', '_')

                                unique_id = f"{setup_id}_{model_name}_{dataset_name}_{seed}_{teacher_pct}"
                                expanded_experiments[unique_id] = config

    return expanded_experiments