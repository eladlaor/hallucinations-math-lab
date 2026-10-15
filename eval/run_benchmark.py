"""Run the configured detectors over a question set and write one JSONL row per question.

Input: JSONL with a `question` field per line (eval/datasets/load_triviaqa.py
output, or eval/datasets/adversarial.jsonl). Ground-truth fields
(`ground_truth_aliases`, `ground_truth_value`, `expected_behavior`) are copied
through untouched so compute_auroc.py can label each row.

Detectors and the escalation rule come from settings (ENABLED_DETECTORS,
ESCALATION_DETECTOR, SE_THRESHOLD); the SLM comes from SLM_PROVIDER /
VLLM_MODEL_ID. Every row records all detector scores under `detector_scores`,
so adding a detector needs no change here or in compute_auroc.py.

Downstream: compute_auroc.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from hallucinations_math_lab import ResponseEvaluator
from hallucinations_math_lab.constants import LOG_EXTRA_KEY

logger = logging.getLogger(__name__)


def run(
    in_path: Path,
    out_path: Path,
    judge: bool = True,
    limit: int | None = None,
    evaluator: ResponseEvaluator | None = None,
) -> None:
    """Score every question in `in_path`. Pass `evaluator` to bring your own
    SamplingProvider, detectors or judge; otherwise one is built from settings."""
    if evaluator is None:
        judge_impl = None
        if judge:
            from hallucinations_math_lab.judges import GeminiJudge

            judge_impl = GeminiJudge()
        evaluator = ResponseEvaluator(judge=judge_impl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with in_path.open() as f_in, out_path.open("w") as f_out:
        for i, line in enumerate(f_in):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            prompt = row["question"]
            try:
                result = evaluator.evaluate(prompt)
            except Exception as e:
                logger.error(
                    "Benchmark row failed; stopping. Rows written so far are kept.",
                    extra={
                        LOG_EXTRA_KEY: {
                            "row_index": i,
                            "row_id": row.get("id"),
                            "rows_written": n_written,
                            "out_path": str(out_path),
                            "error": str(e),
                        }
                    },
                )
                raise

            f_out.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "category": row.get("category"),
                        "question": prompt,
                        "ground_truth": row.get("ground_truth_value")
                        or row.get("expected_behavior"),
                        "ground_truth_aliases": row.get("ground_truth_aliases", []),
                        "greedy_answer": result.greedy_answer,
                        "final_answer": result.final_answer,
                        "final_answer_source": str(result.final_answer_source),
                        "detector_scores": {
                            str(name): score for name, score in result.detector_scores.items()
                        },
                        "escalation": result.escalation_policy_desc
                        or f"{result.escalation_detector} > {result.escalation_threshold}",
                        "escalated": result.escalated,
                        "n_clusters": len(result.clusters) if result.clusters else None,
                        "samples": result.samples,
                    }
                )
                + "\n"
            )
            f_out.flush()
            n_written += 1

    logger.info(
        "Benchmark complete.",
        extra={LOG_EXTRA_KEY: {"rows_written": n_written, "out_path": str(out_path)}},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Run the detectors over a question set.")
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    p.add_argument("--no-judge", action="store_true", help="Skip Gemini escalation.")
    p.add_argument("--limit", type=int, default=None, help="Stop after this many rows.")
    args = p.parse_args()
    run(args.in_path, args.out_path, judge=not args.no_judge, limit=args.limit)
