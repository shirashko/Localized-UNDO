from __future__ import annotations

import asyncio
import json
import math
import multiprocessing
import textwrap
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, Callable, Iterable, List, TypeVar, Any, Tuple, Union, get_args
from datasets import Dataset
import numpy as np
import seaborn as sns
from aiolimiter import AsyncLimiter
from google import genai 
from google.genai import types 
from google.genai.errors import ServerError
import itertools
from numpy import ndarray
from transformers import AutoTokenizer
from localized_undo.utils.paths import DATASET_DIR
import random
import logging
from extract_qa_schemas import PassageID, Probability, PassageQuality, FullPassageQuality, Rating, Response, FullRecord, PROMPT_TYPE, QA, Corpus, generate_instruction, rate_passage_instruction, rate_question_instruction


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

corpuses = get_args(Corpus)

GEMINI_MODEL_NAME = 'gemini-2.0-flash'
T = TypeVar('T')
R = TypeVar('R')

class CorporaSingle(TypedDict):
    text: str
    idx: PassageID

NUM_PROC = multiprocessing.cpu_count()

def mk_gemini_client():
    root_dir = Path(__file__).resolve().parents[2]
    GEMINI_TOKEN_PATH = root_dir / "tokens" / "gemini_token.txt"
    with open(GEMINI_TOKEN_PATH, "r", encoding="utf-8") as f:
        token = f.read().strip()
    client = genai.Client(api_key=token)
    return client


def batched(data: List[Any], batch_size: int, include_leftovers: bool) -> List[Tuple[Any, ...]]:
    """
    Splits a list into smaller batches of a fixed size.
    """
    batches = []
    for i in range(0, len(data), batch_size):
        cur_batch = data[i: i + batch_size]

        # Logic to handle the last batch if it's smaller than batch_size
        if len(cur_batch) < batch_size and not include_leftovers:
            continue

        batches.append(tuple(cur_batch))
    return batches

async def concurrent_single_requests(
    fn: Callable[[T], R],
    arguments: Iterable[T],
    limiter: AsyncLimiter,
    data_len: int
) -> List[R]:
    """
    Executes an asynchronous function concurrently across an iterable of arguments
    while strictly adhering to rate limits defined by an AsyncLimiter.

    This utility orchestrates parallel API calls (e.g., to LLM providers) to maximize
    throughput. It ensures that although execution is unordered and concurrent,
    the resulting list maintains the original input order.

    Args:
        fn: The function to be executed for each item.
            Must accept a single argument of type T and return type R.
        arguments: An iterable of inputs (type T) to be processed by `fn`.
        limiter: An `aiolimiter.AsyncLimiter` instance used to gate the
            execution of `fn` and prevent RateLimitExceeded errors (e.g., HTTP 429).
        data_len: The total number of items in `arguments`. Used to pre-allocate
            the results list for O(1) positional assignment.

    Returns:
        A list of results of type R, where the i-th result corresponds to
        the i-th argument, regardless of the completion order of the tasks.

    Notes:
        - The function uses `asyncio.create_task` to schedule all jobs immediately.
        - The `async with limiter` context manager ensures that the number of
          active requests within a time window does not exceed the quota.
        - If `fn` includes retry logic (e.g., exponential backoff), the limiter
          will continue to gate each individual retry attempt.
    """
    # To follow the result matched to the input arguments
    results = [None] * data_len
    
    async def process_item(item: T, index: int):
        async with limiter:
            result = await fn(item) # The actual Gemini API call is made here
            results[index] = result

    # Create and schedule all tasks (for each argument) immediately
    tasks = [
        asyncio.create_task(process_item(arg, i))
        for i, arg in enumerate(arguments)
    ]
    
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    
    return results


def load_filtered_jsonl(
        file_path: Union[str, Path],
        start_idx: int,
        end_idx: int,
        tokenizer: Any = None,
        min_len: int = 0,
        max_len: float = math.inf
) -> Dataset:
    """
    Selectively loads and filters records from a JSONL file within a specific index range.

    This function facilitates stream-based processing of large-scale corpora (e.g., Wikipedia)
    by reading only a designated slice of the file. It standardizes input formats and
    performs token-level length validation to ensure data quality for LLM tasks.

    Args:
        file_path: Path to the input .jsonl file.
        start_idx: The global line index (inclusive) to start processing from.
        end_idx: The global line index (exclusive) to stop processing at.
        tokenizer: An optional tokenizer (e.g., HuggingFace AutoTokenizer) to calculate
            token-level sequence length.
        min_len: Minimum token count required for a record to be included.
        max_len: Maximum token count allowed for a record to be included.

    Returns:
        A `datasets.Dataset` object containing the validated records, each including
        the original text, its global index (`idx`), and its token length (`len`).

    Notes:
        - The `idx` field preserves the original line number, which is
          essential for mapping generated outputs (like QA) back to source documents.
    """
    filtered_data = []
    with open(file_path, 'r') as f:
        target_lines = itertools.islice(enumerate(f), start_idx, end_idx)

        for i, line in target_lines:
            example = json.loads(line)

            if isinstance(example, str):
                example = {"text": example}

            example['idx'] = i

            if tokenizer is not None:
                example['len'] = len(tokenizer(example["text"])['input_ids'])
                if example['len'] < min_len or example['len'] > max_len:
                    continue

            filtered_data.append(example)

    return Dataset.from_list(filtered_data)


def check_lens():
    tkn = AutoTokenizer.from_pretrained('google/gemma-2-2b')
    results: dict[Corpus, ndarray[int, float]] = {}
    for corpus in corpuses:
        ds = load_filtered_jsonl(f'{DATASET_DIR}/wmdp/{corpus}.jsonl', 0, math.inf)  # Load from local JSONL
        ds = ds.map(lambda x: {"len": [len(s) for s in tkn(x["text"])['input_ids']]}, batched=True, num_proc=NUM_PROC)
        
        results[corpus] = np.percentile(np.array(ds["len"]), np.array([50, 75, 90, 95, 99]))
    return results


async def retry_with_exponential_backoff(
    fn: Callable,
    *args,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.1,
    **kwargs
):
    """
    Retry an async function with exponential backoff.
    
    Args:
        fn: The async function to execute
        *args: Arguments to pass to the function
        max_retries: Maximum number of retries
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Random jitter factor to add to delay
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function
        
    Raises:
        The last exception encountered after all retries are exhausted
    """
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        try:
            return await fn(*args, **kwargs)
        except ServerError as e:
            last_exception = e
            print(e)
            print(dir(e))
            # Check if it's a 503 error
            if hasattr(e, 'code') and e.code == 503:
                retries += 1
                if retries > max_retries:
                    logger.error(f"Max retries exceeded for 503 error: {e}")
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                delay = delay * (1 + random.uniform(-jitter, jitter))
                
                logger.info(f"Model overloaded (503). Retry {retries}/{max_retries} after {delay:.2f}s")
                await asyncio.sleep(delay)
            elif hasattr(e, 'code') and e.code == 429:
                # Resources exhausted, do longer delay because it's per minute
                retries += 1
                logger.info(f"Resources exhausted(429). Retry {retries}/{max_retries} after {30:.2f}s")
                await asyncio.sleep(30)

            else:
                # If it's not a 503 error, re-raise immediately
                raise e
    
    # If we've exhausted retries, raise the last exception
    raise last_exception

async def gemini_quiz_req(
    client: genai.Client, rounds: Sequence[CorporaSingle], *, limiter: AsyncLimiter
) -> list[tuple[PassageID, Response]]:
    async def fn(batch: str) -> Response | None:
        async def api_call():
            return (
                await client.aio.models.generate_content(
                    contents=batch,
                    model=GEMINI_MODEL_NAME,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=Response,
                        system_instruction=generate_instruction,
                    ),
                )
            ).parsed
        return await retry_with_exponential_backoff(api_call)
    res = await concurrent_single_requests(fn, iter([r["text"] for r in rounds]), limiter=limiter, data_len=len(rounds))
    return [(rnd, res) for rnd, res in zip([r["idx"] for r in rounds], res, strict=True) if res is not None]

async def rate_req(client: genai.Client, rounds: Sequence[QA], *, limiter: AsyncLimiter) -> list[Rating]:
    async def fn(batch: QA) -> Rating:
        async def api_call():
            return (
                await client.aio.models.generate_content(
                    contents=batch.question,
                    model=GEMINI_MODEL_NAME,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=Rating,
                        system_instruction=rate_question_instruction,
                    ),
                )
            ).parsed
        return await retry_with_exponential_backoff(api_call)
    return await concurrent_single_requests(fn, iter(rounds), limiter=limiter, data_len=len(rounds))

def run(
    corpus: Corpus, *, batch_size: int, start_idx: int, end_idx: int, limiter: AsyncLimiter, suitability_threshold: float
):
    tkn = AutoTokenizer.from_pretrained('google/gemma-2-2b')
    ds = load_filtered_jsonl(f'{DATASET_DIR}/wmdp/{corpus}.jsonl', start_idx, end_idx, tkn, 200
    , 5000)

    if PROMPT_TYPE == 'cyber':
        suitability_map = mk_suitability_map(corpus)
        ds = ds.filter(
            lambda x: [suitability_map.get(i, 0) > suitability_threshold for i in x["idx"]],
            num_proc=NUM_PROC,
            batched=True,
        )

    client = mk_gemini_client()
    records: list[CorporaSingle] = ds.to_list()
    for b in batched(records, batch_size, include_leftovers=True):
        res = asyncio.run(generate_batch(client, b, limiter=limiter))
        with Path(f"{DATASET_DIR}/wmdp/qa/wmdp-{corpus}-{b[0]['idx']:04}-{b[-1]['idx']:04}.json").open("w") as f:
            print(f"Writing {len(res)} records to {f.name}")
            json.dump([r.model_dump(mode="json") for r in res], f)


def mk_suitability_map(corpus: Corpus) -> Mapping[PassageID, Probability]:
    paths = Path("data").glob(f"wmdp-filter-{corpus}*")
    data = []
    for path in paths:
        with path.open() as f:
            data.extend(json.load(f))
    return {r["id_"]: r["quality"]["is_suitable"] for r in data}


def suitability_histogram():
    with Path(f"{DATASET_DIR}/wmdp/qa/wmdp-filter-cyber-forget-corpus.json").open() as f:
        data = json.load(f)
    suitability: list[float] = [r["quality"]["is_suitable"] for r in data]
    count = 0
    for r in data:
        if r["quality"]["is_suitable"] >= 0.85:
            count += 1
    print(count)
    sns.histplot(np.array(suitability), bins=20)


# After inspection of samples >= 0.85 seems like a reasonable threshold on both forget and retain set


def filter_passages(corpus: Corpus, *, batch_size: int, start_idx: int, end_idx: int, limiter: AsyncLimiter):
    tkn = AutoTokenizer.from_pretrained('google/gemma-2-2b')
    ds = load_filtered_jsonl(f'{DATASET_DIR}/wmdp/{corpus}.jsonl', start_idx, end_idx, tkn, 200, 5000)

    client = mk_gemini_client()
    records: list[CorporaSingle] = ds.to_list()
    for b in batched(records, batch_size, include_leftovers=True):
        res = asyncio.run(filter_batch(client, b, limiter=limiter))
        with Path(f"{DATASET_DIR}/wmdp/qa/wmdp-filter-{corpus}-{b[0]['idx']:04}-{b[-1]['idx']:04}.json").open("w") as f:
            print(f"Writing {len(res)} records to {f.name}")
            json.dump([r.model_dump(mode="json") for r in res], f)


async def generate_batch(client: genai.Client, x: Sequence[CorporaSingle], *, limiter: AsyncLimiter) -> list[FullRecord]:
    passage_qas = await gemini_quiz_req(client, x, limiter=limiter)
    flat_qas = [(pid, qa) for pid, qas in passage_qas for qa in qas.qas]
    # ratings = await rate_req(client, [qa for _, qa in flat_qas], max_concurrency=max_concurrency)
    return [FullRecord(id_=pid, qa=qa) for pid, qa in flat_qas]


async def filter_batch(
    client: genai.Client, rounds: Sequence[CorporaSingle], *, limiter: AsyncLimiter
) -> list[FullPassageQuality]:
    async def fn(batch: str) -> PassageQuality | None:
        # The model may generate invalid JSON and, at temperature 0, retries won't fix that so just return `None`
        return (
            await client.aio.models.generate_content(
                contents=batch,
                model=GEMINI_MODEL_NAME,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=PassageQuality,
                    system_instruction=rate_passage_instruction,
                ),
            )
        ).parsed

    res = await concurrent_single_requests(fn, iter([r["text"] for r in rounds]), limiter=limiter, data_len=len(rounds))
    return [
        FullPassageQuality(id_=idx, quality=res)
        for idx, res in zip([r["idx"] for r in rounds], res, strict=True)
        if res is not None
    ]


def pretty_print_generation(passages: list[str], qass: list[tuple[int, Response]]):
    for p, (_, qas) in zip(passages, qass, strict=True):
        print("-" * 12)
        print(textwrap.fill(p, width=120, replace_whitespace=False))
        for i, qa in enumerate(qas.qas):
            print(f"QA {i + 1}")
            for k, v in qa.model_dump().items():
                print(textwrap.fill(f"{k}: {v}", width=120, replace_whitespace=False))


def pretty_print_ratings(qass: list[tuple[QA, Rating]]):
    for qa, rating in qass:
        print("-" * 12)
        print(textwrap.fill(qa.question, width=120, replace_whitespace=False))
        print(f"Rating: {rating}")
    print(Counter([r for _, r in qass]))


def concat_jsons(corpus):
    paths = Path(f"{DATASET_DIR}/wmdp/qa/").glob(f'{corpus}*')
    print("Combining the following files:")
    data = []
    for path in paths:
        if 'combined' in str(path):
            continue
        print(f"\t{path}")
        with path.open() as f:
            data.extend(json.load(f))

    output_path = f"{DATASET_DIR}/wmdp/qa/{corpus}-combined.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"Saved combined file to {output_path}")


if __name__ == '__main__':

    # concat_jsons('wmdp-bio_remove_dataset')
    # concat_jsons('wmdp-bio_retain_dataset')
    concat_jsons('wmdp-wikipedia')
    # custom_login()
    # step = 500
    # for dataset in ['wikipedia']: # ['bio_retain_dataset', 'bio_remove_dataset']:
    #     for i in range(0, 10000, step):
    #         limiter = AsyncLimiter(max_rate=3, time_period=1)
    #         run(corpus = dataset, batch_size=step, start_idx=i, end_idx=i+step, limiter=limiter, suitability_threshold=None)
