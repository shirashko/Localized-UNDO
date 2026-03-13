import yaml
from localized_undo.utils.paths import MODEL_DIR, DATASET_DIR, CACHE_DIR
import torch


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
                if 'min_lr' in config:
                    config['min_lr'] = float(config['min_lr'])
                else:
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
    Expands a base setup into a list of configurations for alpha/beta/seed sweep,
    incorporating dynamic noise types for localized localization analysis.
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

                # Fetch noise_type (e.g., 'global', 'delta-mask', 'snmf')
                noise_type = config.get('noise_type', 'global')

                for key in ['learning_rate', 'min_lr', 'weight_decay']:
                    if key in config:
                        config[key] = float(config[key])

                # Dynamic Paths
                method = config['method']
                config['teacher_model_name'] = str(MODEL_DIR / config['teacher_rel_path'])
                config['student_model_name'] = config['teacher_model_name']

                # Path Naming: Added noise_type to differentiate localized methods
                path_suffix = f"-{noise_type}-alpha_{alpha}-beta_{beta}-seed_{seed}"
                base_name = f"gemma-2-0.1B_{method}-arithmetic-partial_distill"

                config['output_dir'] = str(MODEL_DIR / "partial_distill_models_arith" / f"{base_name}{path_suffix}")
                config['path_local_record'] = str(
                    MODEL_DIR / "local_records/partial_distill_models_arith" / f"{base_name}{path_suffix}.txt")
                config['wandb_run_name'] = f"{setup_id}_{noise_type}_a{alpha}_b{beta}_s{seed}"

                # Global Data Paths
                config['eng_train_file'] = str(DATASET_DIR / "pretrain/train_eng.jsonl")
                config['arithmetic_train_file'] = str(DATASET_DIR / "pretrain/train_all_arithmetic.jsonl")
                config['eng_valid_file'] = str(DATASET_DIR / "pretrain/valid_eng.jsonl")

                config['cache_dir'] = str(CACHE_DIR)
                config['dataset_cache_dir'] = str(CACHE_DIR)

                mask_path = config.get('noise_mask_rel_path')
                if mask_path:
                    full_mask_path = MODEL_DIR / mask_path
                    # Loading the mask (expecting a Dict of Tensors)
                    config['noise_mask'] = torch.load(full_mask_path, map_location='cpu')
                else:
                    config['noise_mask'] = None

                exp_id = f"{setup_id}_{noise_type}_a{alpha}_b{beta}_s{seed}"
                expanded_experiments[exp_id] = config

    return expanded_experiments

def load_unlearn_configs(yaml_path, base_setup_ids):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    expanded_setups = {}
    for base_id in base_setup_ids:
        # Load defaults then override with specific setup
        config_template = data['default_config'].copy()
        config_template.update(data['setups'][base_id])

        method = config_template['method']
        lr_range = data['lr_ranges'][method]

        for lr in lr_range:
            lr_val = float(lr)
            # Create a unique ID for this specific LR variant
            setup_id = f"{base_id}_lr_{lr_val:.1e}"

            config = config_template.copy()
            config['learning_rate'] = lr_val

            if 'min_lr' in config:
                config['min_lr'] = float(config['min_lr'])

            # Use paths from YAML/Paths.py instead of hard-coded strings
            # Note: We assume these keys exist in your YAML 'setups' or 'default_config'
            config['model_name'] = str(MODEL_DIR / config['model_rel_path'])
            config['forget_train_file'] = str(DATASET_DIR / config['forget_rel_path'])
            config['retain_train_file'] = str(DATASET_DIR / config['retain_rel_path'])
            config['eng_valid_file'] = str(DATASET_DIR / config['valid_rel_path'])

            # Construct output directories dynamically
            method_dir = config['method']
            model_slug = config['model_rel_path'].replace('/', '_')
            config['output_dir'] = str(
                MODEL_DIR / f"unlearned_models/{method_dir}/{model_slug}_lr_{lr_val:.1e}")
            config['path_local_record'] = str(
                MODEL_DIR / f"local_records/unlearned_models/{method_dir}/{model_slug}_lr_{lr_val:.1e}.txt")

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

    return config