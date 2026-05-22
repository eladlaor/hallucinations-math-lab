"""Load and subsample TriviaQA validation set with a fixed seed for reproducibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from taste.constants import RANDOM_SEED


def load_triviaqa_subset(n: int = 1000, seed: int = RANDOM_SEED, out: Path | None = None) -> list[dict]:
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
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
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("eval/datasets/triviaqa_subset.jsonl"))
    args = p.parse_args()
    rows = load_triviaqa_subset(n=args.n, out=args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
