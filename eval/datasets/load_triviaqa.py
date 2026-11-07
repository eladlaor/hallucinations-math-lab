"""Load a reproducible slice of the TriviaQA validation set.

The split lists most questions once per evidence source (17,944 rows, 9,960
questions), so it is first reduced to one row per question. That set is
shuffled once with a fixed seed; `--offset` and `--n` then pick a contiguous
slice of that order. Slices that do not overlap in
`[offset, offset + n)` share no questions, which is how a SEP calibration set
is kept apart from the benchmark it is evaluated on:

    --offset 0   --n 500   -> calibration (probe training)
    --offset 500 --n 1000  -> benchmark

Each row has `id`, `question`, `ground_truth_aliases` and
`ground_truth_value`; run_benchmark.py and collect_sep_dataset.py both read it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from datasets import load_dataset

from hallucinations_math_lab.constants import RANDOM_SEED

# Owner-qualified: current `datasets` releases reject the legacy bare name "trivia_qa".
DATASET_NAME: Final[str] = "mandarjoshi/trivia_qa"
DATASET_CONFIG: Final[str] = "rc.nocontext"
DATASET_SPLIT: Final[str] = "validation"


def slice_bounds(offset: int, n: int, dataset_size: int) -> range:
    """Indices of the requested slice; fails fast if it is empty or out of range."""
    if offset < 0 or n <= 0:
        raise ValueError(f"Need offset >= 0 and n > 0, got offset={offset}, n={n}.")
    if offset + n > dataset_size:
        raise ValueError(
            f"Slice [{offset}, {offset + n}) exceeds the {dataset_size} questions available."
        )
    return range(offset, offset + n)


def unique_question_indices(question_ids: list[str]) -> list[int]:
    """Row indices keeping only the first row of each question, in order."""
    seen: set[str] = set()
    keep: list[int] = []
    for i, qid in enumerate(question_ids):
        if qid not in seen:
            seen.add(qid)
            keep.append(i)
    return keep


def load_triviaqa_subset(
    n: int = 1000, offset: int = 0, seed: int = RANDOM_SEED, out: Path | None = None
) -> list[dict]:
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    ds = ds.select(unique_question_indices(ds["question_id"]))
    ds = ds.shuffle(seed=seed).select(slice_bounds(offset, n, len(ds)))
    rows = [
        {
            "id": ex["question_id"],
            "question": ex["question"],
            "ground_truth_aliases": list(ex["answer"]["aliases"]),
            "ground_truth_value": ex["answer"]["value"],
        }
        for ex in ds
    ]
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=1000, help="Number of questions in the slice.")
    p.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start of the slice in the shuffled order; non-overlapping slices are disjoint.",
    )
    p.add_argument("--out", type=Path, default=Path("eval/datasets/triviaqa_subset.jsonl"))
    args = p.parse_args()
    rows = load_triviaqa_subset(n=args.n, offset=args.offset, out=args.out)
    print(f"Wrote {len(rows)} rows [{args.offset}, {args.offset + len(rows)}) to {args.out}")
