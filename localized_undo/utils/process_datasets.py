from transformers import AutoTokenizer  # For tokenizer
from datasets import Dataset  # For dataset handling


def make_sequence_length(train_ds_list, tokenizer, max_length, join_or_subsequence):
    """
    Standardizes dataset sequence lengths by either packing multiple
    samples into fixed-size windows or filtering out overly long samples.
    """
    if join_or_subsequence:
        def create_exact_len(examples):
            new_input_ids, new_attention_mask = [], []
            cur_input_ids, cur_attn_mask = [], []

            # Identify newline token ID for separation
            newline_token_id = tokenizer("\n")["input_ids"][0]

            for ids, mask in zip(examples["input_ids"], examples["attention_mask"]):
                start_idx = 0
                # Fill the current window until it reaches max_length
                while start_idx < len(ids):
                    remainder = max_length - len(cur_input_ids)
                    cur_input_ids.extend(ids[start_idx: start_idx + remainder])
                    cur_attn_mask.extend(mask[start_idx: start_idx + remainder])
                    start_idx += remainder

                    if len(cur_input_ids) >= max_length:
                        new_input_ids.append(cur_input_ids[:max_length])
                        new_attention_mask.append(cur_attn_mask[:max_length])
                        cur_input_ids = []
                        cur_attn_mask = []

                # Append separator if window is not empty
                cur_input_ids.append(newline_token_id)
                cur_attn_mask.append(1)

            return {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask,
            }

        for i, ds in enumerate(train_ds_list):
            train_ds_list[i] = ds.map(
                create_exact_len,
                batched=True,
                num_proc=4,  # Reduced num_proc as 100 might be too high for some environments
                remove_columns=ds.column_names
            )
        message = f'[process_dataset.py] Created sliding windows of length {max_length}'
    else:
        def filter_long(batch):
            # Keep only samples that fit within the context window
            return [len(ids) <= max_length for ids in batch["input_ids"]]

        # FIX: Calculate total length before filtering using the provided list
        total_before = sum(len(ds) for ds in train_ds_list)

        for i, ds in enumerate(train_ds_list):
            train_ds_list[i] = ds.filter(filter_long, batched=True, batch_size=200_000, num_proc=4)
            # Cleanup to save memory
            if "text" in train_ds_list[i].column_names:
                train_ds_list[i] = train_ds_list[i].remove_columns("text")

        total_after = sum(len(ds) for ds in train_ds_list)
        percent_kept = (total_after / total_before) * 100 if total_before > 0 else 0
        message = f'[process_dataset.py] Filtered for sequence length <= {max_length}. Kept: {percent_kept:.2f}%'

    return train_ds_list, message