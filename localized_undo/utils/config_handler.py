import yaml
from localized_undo.utils.paths import MODEL_DIR, DATASET_DIR, CACHE_DIR, PROJECT_ROOT, WMDP_MODEL_DIR
import os
import re
import json


def _initialize_base_config_by_setup(data, setup_id):
    if 'setups' not in data or setup_id not in data['setups']:
        raise KeyError(f"Setup ID '{setup_id}' not found in the YAML configuration.")

    setup_overrides = data['setups'].get(setup_id) or {}

    return _initialize_base_config(data['default_config'], setup_overrides)


def _initialize_base_config(base_config, overrides):
    config = base_config.copy()
    config.update(overrides)

    config['cache_dir'] = str(CACHE_DIR)
    config['dataset_cache_dir'] = str(CACHE_DIR)

    float_keys = [
        'learning_rate', 'min_lr', 'noise_alpha', 'noise_beta',
        'weight_decay', 'gradient_clipping_threshold',
        'both_losses_act_loss_multiplier', 'use_base_teacher_percent'
    ]
    int_keys = ['batch_size', 'gradient_accumulation_steps', 'max_steps', 'seed', 'max_length', 'epochs']

    for key in float_keys:
        if config.get(key) is not None:
            config[key] = float(config[key])

    for key in int_keys:
        if config.get(key) is not None:
            config[key] = int(config[key])

    return config


def _validate_model_path(model_rel_path, model_v):
    """
    Resolve and validate the model path and its architecture.
    Raises FileNotFoundError or RuntimeError if validation fails.
    """
    full_path = MODEL_DIR / model_rel_path

    # 1. Resolve 'final_model' subfolder if exists
    if (full_path / "final_model").exists():
        full_path = full_path / "final_model"

    # 2. Check existence
    if not full_path.exists():
        raise FileNotFoundError(f"Critical Error: Base model path does not exist: {full_path}")

    # 3. Validate hidden size (Gemma-2 architecture check)
    config_json_path = full_path / "config.json"
    if not config_json_path.exists():
        raise FileNotFoundError(f"Critical Error: Missing 'config.json' in {full_path}. Cannot verify architecture.")

    with open(config_json_path, 'r') as cj:
        model_file_config = json.load(cj)
        actual_hidden = model_file_config.get("hidden_size")

        # Expected mapping for Gemma-2
        expected_hidden = 768 if "0.3B" in model_v else 320

        if actual_hidden and actual_hidden != expected_hidden:
            raise RuntimeError(
                f"Architecture Mismatch! Model at {model_rel_path} has hidden_size={actual_hidden}, "
                f"but metadata expects {model_v} (hidden_size={expected_hidden})."
            )

    return full_path


def _extract_distill_metadata(
    model_dir_name, model_v, method, noise, beta, predefined_distill_noise_label=None
):
    """Extracts alpha (if present) and builds naming for distilled models."""
    alpha_match = re.search(r"alpha_([\d\.]+)", model_dir_name)
    if alpha_match:
        alpha = float(alpha_match.group(1))
        distill_info = f"{model_v}_{method}_{noise}_a{alpha}_b{beta}"
        parent_noise = noise
    elif "predefined-student" in model_dir_name:
        # Student noise is not alpha/beta from this YAML; do not reuse noise_config here.
        alpha = None
        label = predefined_distill_noise_label or "predefined_student"
        distill_info = f"{model_v}_{method}_{label}"
        parent_noise = label
    else:
        raise ValueError(
            f"Could not parse alpha or predefined-student distill from path: {model_dir_name}"
        )

    return {
        'parent_method': method,
        'parent_noise': parent_noise,
        'parent_alpha': alpha,
        'wandb_run_name': f"RL_{distill_info}"
    }


def _extract_baseline_metadata(model_rel_path, model_v):
    """Identifies the type of baseline (Oracle/Unlearn) and builds naming."""
    if "unlearned" in model_rel_path:
        type_label = "Unlearn"
    elif "addition_subtraction+eng" in model_rel_path:
        type_label = "Oracle"
    else:
        raise ValueError(f"Unrecognized baseline type in path: {model_rel_path}")

    return {
        'parent_method': f"baseline_{type_label}",
        'parent_noise': "none",
        'parent_alpha': None,
        'wandb_run_name': f"Base_{model_v}_{type_label}"
    }


def _build_relearn_paths(config, setup_id, model_dir_name, lr_val):
    """Constructs output and local record paths."""
    exp_label = f"relearned_{model_dir_name}_{lr_val:.1e}"
    config['output_dir'] = str(MODEL_DIR / "relearned_models" / setup_id / exp_label)
    config['path_local_record'] = str(
        MODEL_DIR / "local_records/relearned_models" / setup_id / f"{exp_label}.txt"
    )
    return config


def load_relearn_configs(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    meta = data.get('experiment_metadata', {})
    base_template = _initialize_base_config(data['default_config'], meta)
    model_v = meta.get('model_version', 'unknown')
    relearn_lrs = meta.get('relearn_lrs', [])

    base_template['wandb_project'] = f"{model_v}_{base_template['wandb_project']}"

    # gemma-2-0.3B_MaxEnt-arithmetic-partial_distill-arithmetic_gemma-2-0.3B_p50_delta_mask_global-alpha_0.7-beta_0.1-seed_111
    models_to_run = [
        f"partial_distill_models_arith/{model_v}_{meta['method']}-arithmetic-partial_distill-"
        f"{meta['noise_config']}-alpha_{a}-beta_{meta['beta']}-seed_{meta['distill_seed']}"
        for a in meta.get('alphas', [])
    ]

    # e.g. skip_student_corruption distill:
    #   ...-partial_distill-<student_dir_basename>-predefined-student-seed_<seed>
    # or with mask:
    #   ...-partial_distill-<student_dir_basename>-<noise_mask_dir_name>-predefined-student-seed_<seed>
    for rel in meta.get('extra_distilled_models', []) or []:
        rel = rel.strip()
        if not rel:
            continue
        if rel in models_to_run:
            continue
        models_to_run.append(rel)

    if meta.get('include_baselines'):
        baselines = data.get('baselines_library', {}).get(model_v, [])
        models_to_run += (baselines or [])

    expanded_experiments = {}

    for model_rel_path in models_to_run:
        full_model_path = _validate_model_path(model_rel_path, model_v)
        model_dir_name = os.path.basename(model_rel_path)
        is_distilled = "partial_distill_models_arith" in model_rel_path

        for lr in relearn_lrs:
            lr_val = float(lr)
            config = base_template.copy()
            config.update({
                'learning_rate': lr_val,
                'min_lr': lr_val,
                'model_name': str(full_model_path),
                'base_model_version': model_v,
                'eng_valid_file': str(DATASET_DIR / "pretrain/valid_eng.jsonl"),
                'first_train_file': str(DATASET_DIR / config['first_train_file'])
            })

            if is_distilled:
                meta_info = _extract_distill_metadata(
                    model_dir_name,
                    model_v,
                    meta['method'],
                    meta['noise_config'],
                    meta['beta'],
                    predefined_distill_noise_label=meta.get('predefined_distill_noise_label'),
                )
            else:
                meta_info = _extract_baseline_metadata(model_rel_path, model_v)

            meta_info['wandb_run_name'] += f"_lr{lr_val:.1e}"
            config.update(meta_info)

            setup_id = f"{model_v}_train_only_forget"
            config = _build_relearn_paths(config, setup_id, model_dir_name, lr_val)

            unique_id = f"{setup_id}_{model_dir_name}_lr{lr_val}"
            expanded_experiments[unique_id] = config

    print(f"✅ Successfully loaded {len(expanded_experiments)} unique experiments for {model_v}.")
    return expanded_experiments

def load_distill_configs(yaml_path, setup_id):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    base_template = _initialize_base_config_by_setup(data, setup_id)
    model_name_prefix = setup_id.rsplit('_', 1)[0]

    print(f"[*] Config loader: Processing model '{model_name_prefix}' for setup '{setup_id}'")

    if 'stopping_criteria' in data:
        base_template.update(data['stopping_criteria'])

    sweep = data['sweeps']
    expanded_experiments = {}

    # Optional setup behavior:
    # - student_rel_path: initialize student from a custom checkpoint path
    # - skip_student_corruption: disable shrink+perturb/noise entirely
    #   (defaults to True when student_rel_path is explicitly provided)
    custom_student_rel_path = base_template.get('student_rel_path')
    skip_student_corruption = base_template.get('skip_student_corruption', None)
    if skip_student_corruption is None:
        skip_student_corruption = custom_student_rel_path is not None

    if skip_student_corruption:
        alpha_values = [0.0]
        beta_values = [0.0]
    else:
        alpha_values = sweep['alphas']
        beta_values = sweep['betas']

    for alpha in alpha_values:
        for beta in beta_values:
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
                if custom_student_rel_path:
                    student_path = MODEL_DIR / custom_student_rel_path
                    if not student_path.exists():
                        raise FileNotFoundError(f"Student model not found: {student_path}")
                    config['student_model_name'] = str(student_path)
                else:
                    config['student_model_name'] = config['teacher_model_name']

                mask_name = config.get('noise_mask_dir_name')
                if mask_name and not skip_student_corruption:
                    mask_path = PROJECT_ROOT / "localization_masks" / mask_name / "mask.pt"
                    if not mask_path.exists():
                        raise FileNotFoundError(f"Localization mask file missing: {mask_path}")
                    config['noise_mask_path'] = str(mask_path)

                if skip_student_corruption:
                    config['noise_alpha'] = 0.0
                    config['noise_beta'] = 0.0
                    config['shrink_perturb_repeat'] = False
                    config.pop('noise_mask_path', None)
                    config['skip_student_corruption'] = True

                # Experiment Identification & Naming
                mask_config = f"{mask_name}" if mask_name else ""
                if skip_student_corruption:
                    if custom_student_rel_path:
                        student_slug = os.path.basename(
                            os.path.normpath(custom_student_rel_path)
                        )
                    else:
                        student_slug = "from_teacher"
                    if mask_config:
                        exp_id = (
                            f"{setup_id}_{student_slug}_{mask_config}_predefined-student_s{seed}"
                        )
                        path_suffix = (
                            f"-{student_slug}-{mask_config}-predefined-student-seed_{seed}"
                        )
                    else:
                        exp_id = f"{setup_id}_{student_slug}_predefined-student_s{seed}"
                        path_suffix = (
                            f"-{student_slug}-predefined-student-seed_{seed}"
                        )
                else:
                    exp_id = f"{setup_id}_{mask_config}_a{float(alpha)}_b{float(beta)}_s{seed}"
                    path_suffix = f"-{mask_config}-alpha_{alpha}-beta_{beta}-seed_{seed}"

                base_name = f"{model_name_prefix}_{method}-arithmetic-partial_distill"

                config['output_dir'] = str(MODEL_DIR / "partial_distill_models_arith" / f"{base_name}{path_suffix}")
                config['path_local_record'] = str(
                    MODEL_DIR / "local_records/partial_distill_models_arith" / f"{base_name}{path_suffix}.txt")
                config['wandb_run_name'] = exp_id

                # Data Paths
                config['eng_train_file'] = str(DATASET_DIR / "pretrain/train_eng.jsonl")
                config['arithmetic_train_file'] = str(DATASET_DIR / "pretrain/train_all_arithmetic.jsonl")
                config['eng_valid_file'] = str(DATASET_DIR / "pretrain/valid_eng.jsonl")


                expanded_experiments[exp_id] = config

    return expanded_experiments


def load_unlearn_configs(yaml_path, base_setup_ids):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    expanded_setups = {}
    for base_id in base_setup_ids:
        config_template = _initialize_base_config_by_setup(data, base_id)
        method = config_template['method']
        lr_range = data['lr_ranges'][method]

        for lr in lr_range:
            lr_val = float(lr)
            setup_id = f"{base_id}_lr_{lr_val:.1e}"

            config = config_template.copy()
            config['learning_rate'] = lr_val
            config['min_lr'] = lr_val

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

    config = _initialize_base_config_by_setup(data, setup_id)

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
        config_template = _initialize_base_config_by_setup(data, base_id)

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
        config_template = _initialize_base_config_by_setup(data, setup_id)

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


def load_wmdp_relearn_configs(yaml_path, setup_ids, model_map, eval_on_loss=False):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    sweep_seeds = data.get('sweeps', {}).get('seeds', [42])
    dataset_info = data.get('datasets', {})
    expanded_experiments = {}

    for setup_id in setup_ids:
        config_template = _initialize_base_config_by_setup(data, setup_id)

        for model_name, model_path_template in model_map.items():
            for data_name, d_val in dataset_info.items():

                # Domain safety check: Only pair bio data with bio models, etc.
                if not (("bio" in data_name and "bio" in model_name) or
                        ("cyber" in data_name and "cyber" in model_name)):
                    continue

                for seed in sweep_seeds:
                    for lr in d_val['lrs']:
                        config = config_template.copy()

                        # Handle training logic
                        config['learning_rate'] = float(lr)
                        config['min_lr'] = float(lr)
                        config['seed'] = int(seed)
                        config['model_name'] = str(WMDP_MODEL_DIR / model_path_template.replace("SEED", str(seed)))

                        # Dataset setup
                        config['train_files'] = [str(DATASET_DIR / f) for f in d_val['files']]
                        config['interleave_probs'] = d_val['probs']

                        # Dynamic naming logic
                        eval_prefix = 'loss_eval/' if eval_on_loss else ''
                        params_slug = f"{eval_prefix}data-{data_name}_lr-{lr}_model-{model_name}_seed-{seed}"

                        config['output_dir'] = str(WMDP_MODEL_DIR / "relearned_models" / params_slug)
                        config['path_local_record'] = str(
                            WMDP_MODEL_DIR / "local_records" / "relearned_models" / f"{params_slug}.txt")
                        config['wandb_run_name'] = params_slug.replace('/', '_')
                        config['wandb_project'] = f"relearn_{'cyber' if 'cyber' in model_name else 'bio'}"

                        # Special case for initial evaluations
                        if 'initial' in data_name:
                            config['validation_steps'] = [0]
                            config['max_steps'] = 1

                        unique_id = f"{setup_id}_{params_slug}"
                        expanded_experiments[unique_id] = config

    return expanded_experiments