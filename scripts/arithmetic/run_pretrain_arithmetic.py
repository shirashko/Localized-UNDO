import argparse
from accelerate import Accelerator
from localized_undo.tools.pretrain import train
from localized_undo.utils.paths import WANDB_API_KEY_PATH, CONFIG_DIR
from localized_undo.utils.config_handler import load_pretrain_config
from localized_undo.utils.validation_functions import get_arithmetic_eval_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=str, required=True, help="setup_id from YAML")
    args = parser.parse_args()

    # 1. Load configuration
    yaml_path = CONFIG_DIR / "arithmetic" / "pretrain.yaml"
    config = load_pretrain_config(yaml_path, args.setup)

    # 2. Load WandB API key
    try:
        api_key = WANDB_API_KEY_PATH.read_text().strip()
    except Exception as e:
        print(f"Error reading WandB key: {e}")
        return

    accelerator = Accelerator()

    # 3. Create the evaluation function
    eval_fn = get_arithmetic_eval_fn(
        model_name=config['model_name'],
        eng_valid_file=config['eng_valid_file'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        cache_dir=config['cache_dir'],
        dataset_cache_dir=config['dataset_cache_dir'],
        num_wiki_batches=50,
        accelerator=accelerator
    )

    # 4. Prepare training files (The flattening fix)
    train_files = [config['eng_train_file']] + config['secondary_train_files']

    # 5. Filter parameters to match the train() function signature
    # We remove keys that are used for setup/loading but not by the train function itself
    keys_to_exclude = [
        'model_id',
        'arithmetic_type',
        'eng_train_file',
        'secondary_train_files',
        'eng_valid_file'
    ]

    train_params = {k: v for k, v in config.items() if k not in keys_to_exclude}

    # 6. Run the training loop
    train(
        eval_fn=eval_fn,
        accelerator=accelerator,
        wandb_api_key=api_key,
        train_files=train_files,
        **train_params
    )


if __name__ == "__main__":
    main()