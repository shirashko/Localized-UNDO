#!/usr/bin/env python3
"""
Download raw WMDP **bio** forget / retain corpora into ``datasets/wmdp/``.

These JSONL files are the inputs expected by ``localized_undo/prepare_data/prepare.py``,
which builds the tokenized relearn paths used in ``run_relearn_wmdp.py``, e.g.:

  - ``datasets/pretrain/train_bio_remove_dataset.jsonl``  (forget)
  - ``datasets/pretrain/train_bio_retain_dataset.jsonl``  (retain)

Sources (CAIS WMDP benchmark — see https://github.com/centerforaisafety/wmdp):

  - **Forget (remove):** ``cais/wmdp-bio-forget-corpus`` — gated on Hugging Face. you must
    accept the dataset terms and use a token with access (``tokens/hf_token.txt``).
  - **Retain:** ``cais/wmdp-corpora`` config ``bio-retain-corpus`` — public.

After this script succeeds, run from repo root:
    python localized_undo/prepare_data/prepare.py
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
from localized_undo.utils.paths import DATASET_DIR, PROJECT_ROOT

try:
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit(
        "The `datasets` package is required. Install project deps (e.g. `uv sync`)."
    ) from e


FORGET_DATASET = "cais/wmdp-bio-forget-corpus"
RETAIN_DATASET = "cais/wmdp-corpora"
RETAIN_CONFIG = "bio-retain-corpus"

FORGET_OUT = "bio_remove_dataset.jsonl"
RETAIN_OUT = "bio_retain_dataset.jsonl"


def _read_hf_token() -> str | None:
    path = PROJECT_ROOT / "tokens" / "hf_token.txt"
    if not path.is_file():
        return None
    return path.read_text().strip() or None


def _row_to_jsonl_line(row: dict) -> str:
    if "text" not in row:
        raise KeyError(f"Expected a 'text' field in row, got keys: {list(row.keys())}")
    return json.dumps({"text": row["text"]}, ensure_ascii=False)


def stream_dataset_to_jsonl(
    *,
    out_path: Path,
    load_kw: dict,
    desc: str,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[skip] Already exists: {out_path}", flush=True)
        return 0

    ds = load_dataset(**load_kw, streaming=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc=desc, mininterval=2):
            f.write(_row_to_jsonl_line(row) + "\n")
            count += 1
    print(f"[done] Wrote {count} lines -> {out_path}", flush=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download WMDP bio forget/retain corpora."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATASET_DIR / "wmdp",
        help="Directory for bio_remove_dataset.jsonl and bio_retain_dataset.jsonl",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (default: read tokens/hf_token.txt if present)",
    )
    args = parser.parse_args()

    token = args.token or _read_hf_token()
    out_dir = args.out_dir.resolve()

    stream_dataset_to_jsonl(
        out_path=out_dir / RETAIN_OUT,
        load_kw={
            "path": RETAIN_DATASET,
            "name": RETAIN_CONFIG,
            "split": "train",
        },
        desc=f"{RETAIN_DATASET}/{RETAIN_CONFIG}",
    )

    if not token:
        print(
            "[error] Bio forget corpus is gated. Set --token or create tokens/hf_token.txt "
            "with a Hugging Face token that has access to "
            f"https://huggingface.co/datasets/{FORGET_DATASET}",
            file=sys.stderr,
        )
        sys.exit(1)
    stream_dataset_to_jsonl(
        out_path=out_dir / FORGET_OUT,
        load_kw={
            "path": FORGET_DATASET,
            "split": "train",
            "token": token,
        },
        desc=FORGET_DATASET,
    )

    print(
        "\nNext: run `python localized_undo/prepare_data/prepare.py` to build "
        "`datasets/pretrain/train_bio_remove_dataset.jsonl` and "
        "`train_bio_retain_dataset.jsonl`.",
        flush=True,
    )


if __name__ == "__main__":
    main()
