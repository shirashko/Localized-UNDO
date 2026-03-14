import os
import random
import math
import json
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding
)
import wandb
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from localized_undo.utils.loss_functions import cross_entropy_loss_fn, print_acc, custom_makedirs
from localized_undo.utils.process_datasets import make_sequence_length

def relearn(
    model_name,

    train_files,
    eval_fn,
    accelerator,
    join_or_subsequence,
    interleave_probs,
    output_dir,
    cache_dir,
    dataset_cache_dir,

    seed=42,
    device="cuda",
    batch_size=4,
    gradient_accumulation_steps=16,
    epochs=2,
    learning_rate=4e-4,
    max_steps=-1,
    num_warmup_steps=100,
    validation_steps=50,
    save_checkpoint_steps=1500,
    scheduler_type="cosine",
    min_lr=4e-5,
    weight_decay=0.1,
    gradient_clipping_threshold=1.0,
    max_length=2048,

    use_wandb=False,
    wandb_project="gemma-2-0.1B-relearn",
    wandb_run_name=None,

    use_local_record=True,
    path_local_record="local_record/relearn_log.jsonl",
    stopping_strategy='first_exhausted',
    overwrite_ok=False,
    save_models=True
):
    """
    Executes a controlled training loop to "re-acquire" knowledge in a model.

    This function is designed to measure how efficiently a model can be fine-tuned
    on specific data. It supports multi-source data interleaving and intensive
    validation logging (Cross-Entropy and Perplexity) to monitor the recovery of
    specific capabilities.

    Args:
        model_name (str): Path to the local model directory or Hugging Face model ID.
        train_files (list[str]): List of paths to JSONL training data files.
        eval_fn (callable): A validation function that accepts the model and
            returns a dictionary of metrics (e.g., accuracy, loss).
        accelerator (accelerate.Accelerator): Initialized Accelerator for distributed training.
        join_or_subsequence (bool): If True, joins short examples to reach max_length.
            If False, filters out examples exceeding max_length.
        interleave_probs (list[float]): Sampling probabilities for each source in train_files.
        output_dir (str): Directory where final models and checkpoints will be saved if save_models is True.
        cache_dir (str): Directory for model/tokenizer binary caching.
        dataset_cache_dir (str): Directory for processed dataset caching.
        seed (int): Random seed for reproducibility.
        device (str): Computation device (default: "cuda").
        batch_size (int): Training batch size per device.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before an optimizer update.
        epochs (int): Number of full passes over the interleaved dataset.
        learning_rate (float): Peak learning rate for the scheduler.
        max_steps (int): Total training steps. If > 0, overrides epoch-based duration.
        num_warmup_steps (int): Steps for the learning rate warmup phase.
        validation_steps (int): Frequency of evaluation during training.
        save_checkpoint_steps (int): Frequency of saving intermediate model weights if save_models is True.
        scheduler_type (str): Type of LR scheduler ("linear" or "cosine").
        min_lr (float): The floor for the learning rate decay.
        weight_decay (float): Weight decay coefficient for AdamW.
        gradient_clipping_threshold (float): Max norm for gradient clipping.
        max_length (int): Maximum sequence length for tokenization.
        use_wandb (bool): Enable Weights & Biases logging.
        wandb_project (str): W&B project name.
        wandb_run_name (str): Specific W&B run name.
        use_local_record (bool): If True, saves JSONL logs to path_local_record.
        path_local_record (str): File path for local JSONL logging.
        stopping_strategy (str): Data interleaving strategy ('first_exhausted' or 'all_exhausted').
        overwrite_ok (bool): If True, allows writing to an existing output_dir.
        save_models (bool): If True, saves the final model and checkpoints.

    Note:
        The function performs an initial validation step at step 0 to establish a baseline
        and a final validation step upon completion or early stopping.
    """
    # --------------------------------------------------
    # Accelerator, logging, seeds
    # --------------------------------------------------
    print_message = accelerator.is_main_process
    torch.set_default_dtype(torch.bfloat16)

    args_dict = {**locals()}
    print_acc(f"[relearn.py] Initiated relearn with:\n{args_dict}", print_message)

    custom_makedirs(output_dir, exist_ok=overwrite_ok)
    random.seed(seed)
    torch.manual_seed(seed)

    # W&B initialization
    if use_wandb and accelerator.is_main_process:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=args_dict,
        )

    # Local record setup
    if use_local_record and accelerator.is_main_process:
        custom_makedirs(path_local_record, exist_ok=overwrite_ok)

    # --------------------------------------------------
    # 1) Load model + tokenizer
    # --------------------------------------------------
    print_acc(f"[relearn.py] Loading model {model_name}", print_message)
    model_config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        attn_implementation='eager',
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ----------------------------------------------------------------
    # 2. Load training datasets
    # ----------------------------------------------------------------
    train_ds_list = []
    for file in train_files:
        print_acc(f"[pretrain.py] Loading train dataset from {file}", print_message)
        train_ds = load_dataset("json", data_files=file, split="train", cache_dir=dataset_cache_dir)
        train_ds_list.append(train_ds)
   
    # If join_or_subsequence, form sequences of exactly max_length 
    # by joining multiple or using subsequences
    # else filter for only those less than max_length
    train_ds_list, message = make_sequence_length(train_ds_list=train_ds_list, tokenizer=tokenizer, max_length=max_length, join_or_subsequence=join_or_subsequence)
    print_acc(message, print_message)
    
    # If both were given, interleave with equal probabilities
    # If only one is present, that's our train_ds
    if len(train_ds_list) == 0:
        raise ValueError("No training datasets were provided!")
    elif len(train_ds_list) == 1:
        train_ds = train_ds_list[0]
    else:
        lengths = [len(item) for item in train_ds_list]
        print_acc(f"[pretrain.py] Interleave: {interleave_probs} for datasets with lengths {lengths}", print_message)
        train_ds = interleave_datasets(train_ds_list, probabilities=interleave_probs, seed=seed, stopping_strategy=stopping_strategy)

    print_acc(f"[pretrain.py] Train dataset size: {len(train_ds)}", print_message)

   
    # --------------------------------------------------
    # 3) Create DataLoaders
    # --------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    
    steps_per_epoch = len(train_loader)
    effective_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
    total_steps = effective_steps_per_epoch * epochs
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    print_acc(f"[relearn.py] {steps_per_epoch} steps per epoch, {total_steps // accelerator.num_processes} steps total", print_message)

    # --------------------------------------------------
    # 4) Optimizer / LR scheduler
    # --------------------------------------------------
    print_acc(f"[relearn.py] Using AdamW optimizer, LR={learning_rate}, weight_decay={weight_decay}", print_message)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if scheduler_type == "linear":
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
                return (1.0 - progress) * (1.0 - (min_lr / learning_rate)) + (min_lr / learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == "cosine":
        def cosine_lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return cosine_decay * (1.0 - (min_lr / learning_rate)) + (min_lr / learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    # --------------------------------------------------
    # 5) Prepare with Accelerator
    # --------------------------------------------------
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # --------------------------------------------------
    # (NEW) Initial Validation Step
    # --------------------------------------------------
    print_acc("[relearn.py] Running initial validation before training", print_message)

    
    initial_val_log_dict = eval_fn(model, print_results=True)
    initial_val_log_dict["train/step"] = 0
    initial_val_log_dict["train/tokens_seen"] = 0

    if use_wandb and accelerator.is_main_process:
        wandb.log(initial_val_log_dict)
    if use_local_record and accelerator.is_main_process:
        with open(path_local_record, "a", encoding="utf-8") as f:
            f.write(json.dumps(initial_val_log_dict) + "\n")

    # --------------------------------------------------
    # 6) Training Loop
    # --------------------------------------------------
    print_acc("[relearn.py] Starting training", print_message)
    global_step = 0
    global_tokens = 0
    stop_early = False

    for epoch in range(epochs):
        print_acc(f"[relearn.py] Epoch {epoch+1}/{epochs}", print_message)
        model.train()

        for step_in_epoch, batch in enumerate(train_loader):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            ce_loss = cross_entropy_loss_fn(outputs.logits, batch["input_ids"], tokenizer.pad_token_id)

            # gradient accumulation
            loss = ce_loss / gradient_accumulation_steps
            accelerator.backward(loss)

            # count tokens
            tokens_this_batch = batch["attention_mask"].sum().detach()
            tokens_this_batch = accelerator.gather(tokens_this_batch).sum().item()
            global_tokens += tokens_this_batch

            # Update
            if (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), gradient_clipping_threshold)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Print debug info
                if global_step == 1 or global_step % 5 == 0:
                    print_acc(
                        f"[relearn.py] Epoch {epoch+1}/{epochs}, "
                        f"Step {global_step}/{total_steps // accelerator.num_processes}, "
                        f"CE loss: {ce_loss:.6f}",
                        print_message
                    )

                # Logging
                train_log_dict = {
                    "train/ce_loss": ce_loss.item(),
                    "train/step": global_step,
                    "train/tokens_seen": global_tokens,
                    "train/lr": scheduler.get_last_lr()[0],
                }

                if use_wandb and accelerator.is_main_process:
                    wandb.log(train_log_dict)
                if use_local_record and accelerator.is_main_process:
                    with open(path_local_record, "a", encoding="utf-8") as f:
                        f.write(json.dumps(train_log_dict) + "\n")

                # Validation
                if (
                    global_step % validation_steps == 0 or max_steps > 0 and global_step >= max_steps
                    or step_in_epoch == len(train_loader) - 1 # eval at end of epoch
                    #or global_step < 100 # added for relearning
                ):
                    print_acc("[relearn.py] Running validation ...", print_message)

                    val_log_dict = eval_fn(model, print_results=True)
                    val_log_dict["train/step"] = global_step
                    val_log_dict["train/tokens_seen"] = global_tokens
                    if use_wandb and accelerator.is_main_process:
                        wandb.log(val_log_dict)
                    if use_local_record and accelerator.is_main_process:
                        with open(path_local_record, "a", encoding="utf-8") as f:
                            f.write(json.dumps(val_log_dict) + "\n")

                # Checkpoint
                if save_models and save_checkpoint_steps > 0 and (global_step % save_checkpoint_steps == 0):
                    if accelerator.is_main_process:
                        checkpoint_path = os.path.join(output_dir, f"checkpoint-step{global_step}")
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(checkpoint_path)
                        tokenizer.save_pretrained(checkpoint_path)
                        print_acc(f"[relearn.py] Saved checkpoint => {checkpoint_path}", print_message)

                # Early stopping
                if max_steps > 0 and global_step >= max_steps:
                    print_acc("[relearn.py] Reached max_steps => Stopping.", print_message)
                    stop_early = True
                    break

        if stop_early:
            break

    # --------------------------------------------------
    # Final Save
    # --------------------------------------------------
    if save_models and accelerator.is_main_process:
        model.eval()
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = os.path.join(output_dir, "final_model")
        unwrapped_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print_acc(f"[relearn.py] Model saved to => {save_path}", print_message)