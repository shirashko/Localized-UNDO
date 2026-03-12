import os
import orjson
import argparse
import sys
from datasets import load_dataset
from tqdm import tqdm

# Setup paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from localized_undo.utils.paths import CACHE_DIR, DATASET_DIR

OUTPUT_DIR = DATASET_DIR / "fineweb"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_dataset(dataset_id, subset_name, output_filename, split="train", cache_dir=CACHE_DIR, batch_size=1_000_000,
                     max_rows=None):
    print(f"\n[download_datasets.py] Starting subset '{subset_name}' from '{dataset_id}'...")

    try:
        if subset_name is None:
            ds = load_dataset(dataset_id, split=split, streaming=True, cache_dir=cache_dir)
        else:
            ds = load_dataset(dataset_id, name=subset_name, split=split, streaming=True, cache_dir=cache_dir)

        total = ds.num_rows if hasattr(ds, "num_rows") and ds.num_rows is not None else None
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        buffer = []
        count = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in tqdm(ds, total=total, desc=f"Processing {output_filename}", mininterval=5):
                buffer.append(orjson.dumps(sample).decode("utf-8"))
                count += 1

                if count % batch_size == 0:
                    f.write("\n".join(buffer) + "\n")
                    print(f"  > Processed {count} lines so far...")
                    buffer.clear()

                if max_rows is not None and count >= max_rows:
                    break

            if buffer:
                f.write("\n".join(buffer) + "\n")

        print(f"[Finished] Streamed {count} lines to: {output_path}")

    except Exception as e:
        print(f"[Error] Failed to download {dataset_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for specific UNDO settings.")
    parser.add_argument(
        "--mode",
        type=str,
        default="arithmetic",
        choices=["all", "language", "arithmetic", "wmdp"],
        help="Which task setting to download data for. Default is 'arithmetic'."
    )
    args = parser.parse_args()

    print(f"--- Running in '{args.mode}' mode ---")

    # Determine which groups to run
    do_wmdp = args.mode in ["all", "wmdp"]
    do_language = args.mode in ["all", "language"]
    do_arithmetic = args.mode in ["all", "arithmetic"]

    # 1. WMDP EXCLUSIVE DATASETS
    if do_wmdp:
        wmdp_list = [
            ("Magpie-Align/Magpie-Llama-3.3-Pro-1M-v0.1", None, "magpie.jsonl"),
            ("Magpie-Align/Magpie-Llama-3.1-Pro-1M-v0.1", None, "magpie3-1.jsonl"),
            ("Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1", None, "magpie-qwen.jsonl"),
            ("Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1", None, "magpie-3.jsonl"),
            ("Magpie-Align/Magpie-Qwen2-Pro-1M-v0.1", None, "magpie-qwen2.jsonl"),
            ("Magpie-Align/Magpie-Phi3-Pro-1M-v0.1", None, "magpie-phi3.jsonl"),
            ("Magpie-Align/Magpie-Gemma2-Pro-534K-v0.1", None, "magpie-gemma2.jsonl"),
            ("Magpie-Align/Magpie-Pro-300K-Filtered", None, "magpie-filtered.jsonl"),
            ("Salesforce/wikitext", "wikitext-2-v1", "wikitext.jsonl"),
            ("legacy-datasets/wikipedia", "20220301.en", "wikipedia.jsonl"),
        ]
        for d_id, sub, out in wmdp_list:
            download_dataset(d_id, sub, out, max_rows=10_000_000)

    # 2. LANGUAGE EXCLUSIVE DATASETS
    if do_language:
        download_dataset(
            dataset_id="HuggingFaceFW/fineweb-2",
            subset_name="kor_Hang",
            output_filename="fineweb2_kor.jsonl",
            max_rows=10_000_000
        )

    # 3. SHARED DATASETS (Required for Language, Arithmetic, and WMDP)
    if do_arithmetic or do_language or do_wmdp:
        download_dataset(
            dataset_id="HuggingFaceFW/fineweb-edu",
            subset_name="sample-10BT",
            output_filename="fineweb_eng_sample-10BT.jsonl",
            max_rows=10_000_000
        )


if __name__ == "__main__":
    main()