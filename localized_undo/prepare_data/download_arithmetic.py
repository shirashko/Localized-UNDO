import os
import orjson
import sys
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from localized_undo.utils.paths import DATASET_DIR
from localized_undo.utils.generate_arithmetic import get_equations, get_template_word_problems

ARITH_DIR = DATASET_DIR / "arithmetic"
SEED = 99


def generate_and_save(operations, samples_per_type, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Calculate total for clearer logging
    total_expected = samples_per_type * 2
    print(f"\nGenerating {total_expected} total examples ({samples_per_type} equations + {samples_per_type} word problems) for: {operations}")

    # Generate equations and word problems
    equation_samples = get_equations(operations=operations, seed=SEED, amount=samples_per_type, val=False)
    word_problem_samples = get_template_word_problems(operations=operations, seed=SEED, amount=samples_per_type, val=False)

    # Shuffle everything together
    random.seed(SEED)
    all_elements = list(equation_samples + word_problem_samples)
    random.shuffle(all_elements)

    # Save to JSONL format
    with open(output_path, "wb") as f:
        for i, element in enumerate(all_elements):
            data_dict = {"text": element}
            serialized_data = orjson.dumps(data_dict) + b"\n"
            f.write(serialized_data)

            # Progress tracking: print every 100,000 lines to avoid log flooding
            if (i + 1) % 100000 == 0:
                print(f"  > Progress: {i + 1} / {len(all_elements)} lines written...")

    print(f"Successfully finished. Saved to: {output_path}")

# Generate the three required arithmetic datasets
generate_and_save(
    operations=['addition', 'subtraction', 'multiplication', 'division'],
    samples_per_type=1_000_000,
    output_path=str(ARITH_DIR / 'all_arithmetic.jsonl')
)

generate_and_save(
    operations=['addition', 'subtraction'],
    samples_per_type=500_000,
    output_path=str(ARITH_DIR / 'addition_subtraction.jsonl')
)

generate_and_save(
    operations=['multiplication', 'division'],
    samples_per_type=500_000,
    output_path=str(ARITH_DIR / 'multiplication_division.jsonl')
)