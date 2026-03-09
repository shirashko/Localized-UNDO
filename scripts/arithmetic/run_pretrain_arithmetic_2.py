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

    # load configuration
    yaml_path = CONFIG_DIR / "arithmetic" / "pretrain.yaml"
    config = load_pretrain_config(yaml_path, args.setup)

    # load WandB API key
    try:
        api_key = WANDB_API_KEY_PATH.read_text().strip()
    except Exception as e:
        print(f"Error reading WandB key: {e} with exception {e}")
        return

    accelerator = Accelerator()

    # Create the evaluation function for arithmetic tasks
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

    # Running the training loop
    train_params = {k: v for k, v in config.items() if k not in ['model_id', 'arithmetic_type']}

    train_files = [config['eng_train_file']] + config['secondary_train_files']

    train(
        eval_fn=eval_fn,
        accelerator=accelerator,
        wandb_api_key=api_key,
        train_files=train_files,
        **train_params
    )


if __name__ == "__main__":
    main()