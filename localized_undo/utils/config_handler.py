import yaml
from localized_undo.utils.paths import MODEL_DIR, DATASET_DIR, CACHE_DIR


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