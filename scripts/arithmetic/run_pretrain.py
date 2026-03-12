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

    # 4. Prepare training files
    train_files = [config['eng_train_file']] + config['secondary_train_files']

    # 5. Run the training loop
    train(
        # Model & Paths
        model_name=config['model_name'],
        train_files=train_files,
        interleave_probs=config['interleave_probs'],
        output_dir=config['output_dir'],
        cache_dir=config['cache_dir'],
        dataset_cache_dir=config['dataset_cache_dir'],

        # Validation & Accelerator
        eval_fn=eval_fn,
        validation_steps=config['validation_steps'],
        accelerator=accelerator,

        # Training Hyperparameters
        seed=config['seed'],
        device=config['device'],
        batch_size=config['batch_size'],
        join_or_subsequence=config['join_or_subsequence'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        max_steps=config['max_steps'],
        num_warmup_steps=config['num_warmup_steps'],
        save_checkpoint_steps=config['save_checkpoint_steps'],
        scheduler_type=config['scheduler_type'],
        min_lr=config['min_lr'],
        weight_decay=config['weight_decay'],
        gradient_clipping_threshold=config['gradient_clipping_threshold'],
        max_length=config['max_length'],

        # WandB
        use_wandb=config['use_wandb'],
        wandb_project=config['wandb_project'],
        wandb_run_name=config['wandb_run_name'],
        wandb_api_key=api_key,

        # Logging
        use_local_record=config['use_local_record'],
        path_local_record=config['path_local_record']
    )

if __name__ == "__main__":
    main()