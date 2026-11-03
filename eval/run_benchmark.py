"""Run the configured detectors over a question set and write one JSONL row per question.

Input: JSONL with a `question` field per line (eval/datasets/load_triviaqa.py
output, or eval/datasets/adversarial.jsonl). Ground-truth fields
(`ground_truth_aliases`, `ground_truth_value`, `expected_behavior`) are copied
through untouched so compute_auroc.py can label each row.

Detectors and the escalation rule come from settings (ENABLED_DETECTORS,
ESCALATION_DETECTOR, SE_THRESHOLD); the SLM comes from SLM_PROVIDER /
VLLM_MODEL_ID. Every row records all detector scores under `detector_scores`,
so adding a detector needs no change here or in compute_auroc.py.

With a judge, every row records the judge's answer as `judge_answer`, not
only the escalated rows, so escalation_uplift.py can compare the detector's
escalations with random ones at the same rate. This costs one judge call per
question.

Every row also records the run's `model_id`, `sample_n`,
`sample_temperature`, `max_tokens` and `answer_instruction` (the brief-answer
prompt); the scoring scripts refuse to mix rows from different runs or to
compare runs that differ in them.

When a Semantic Entropy Probe is among the detectors, the run fails before
scoring anything if a question was in the probe's training set (see
eval/sep_probe_provenance.py).

Downstream: compute_auroc.py, escalation_uplift.py, adversarial_separation.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from eval.benchmark_results import (
    ANSWER_INSTRUCTION_FIELD,
    ID_FIELD,
    JUDGE_ANSWER_FIELD,
    MAX_TOKENS_FIELD,
    MODEL_ID_FIELD,
    SAMPLE_N_FIELD,
    SAMPLE_TEMPERATURE_FIELD,
    RunConfig,
)
from eval.sep_probe_provenance import assert_disjoint
from hallucinations_math_lab import ResponseEvaluator
from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import LOG_EXTRA_KEY, AnswerSource
from hallucinations_math_lab.custom_types.domain_types import DetectionResult
from hallucinations_math_lab.services.semantic_entropy_probe import SemanticEntropyProbeDetector

logger = logging.getLogger(__name__)


def _read_questions(in_path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with in_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and len(rows) >= limit:
                break
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{in_path}:{line_no} is not valid JSON: {e}") from e
    return rows


def guard_probe_leakage(evaluator: ResponseEvaluator, questions: list[dict[str, Any]]) -> None:
    """Fail fast if any SEP probe in use was trained on one of these questions."""
    ids = [str(q[ID_FIELD]) for q in questions if q.get(ID_FIELD) is not None]
    for detector in evaluator.detectors:
        if isinstance(detector, SemanticEntropyProbeDetector):
            assert_disjoint(detector.probe_path, ids)


def _judge_answer(evaluator: ResponseEvaluator, prompt: str, result: DetectionResult) -> str | None:
    """The judge's answer to `prompt`, for every row, or None without a judge.

    Escalated rows reuse the answer the judge already gave; the rest ask the
    judge once more. escalation_uplift.py needs it on every row to compare
    the detector's escalations with random ones at the same rate.
    """
    if result.final_answer_source == AnswerSource.JUDGE_MODEL:
        return result.final_answer
    if evaluator.judge is None:
        return None
    answer, _reasoning = evaluator.judge.answer(prompt)
    return answer


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

    questions = _read_questions(in_path, limit)
    guard_probe_leakage(evaluator, questions)
    settings = get_settings()
    run_config: RunConfig = {
        MODEL_ID_FIELD: evaluator.slm.model_id,
        SAMPLE_N_FIELD: settings.sample_n,
        SAMPLE_TEMPERATURE_FIELD: settings.sample_temperature,
        MAX_TOKENS_FIELD: settings.max_tokens,
        ANSWER_INSTRUCTION_FIELD: settings.answer_instruction,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with out_path.open("w") as f_out:
        for i, row in enumerate(questions):
            prompt = row["question"]
            try:
                result = evaluator.evaluate(prompt)
                judge_answer = _judge_answer(evaluator, prompt, result)
            except Exception as e:
                logger.error(
                    "Benchmark row failed; stopping. Rows written so far are kept.",
                    extra={
                        LOG_EXTRA_KEY: {
                            "row_index": i,
                            "row_id": row.get(ID_FIELD),
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
                        ID_FIELD: row.get(ID_FIELD),
                        "category": row.get("category"),
                        "question": prompt,
                        "ground_truth": row.get("ground_truth_value")
                        or row.get("expected_behavior"),
                        "ground_truth_aliases": row.get("ground_truth_aliases", []),
                        **run_config,
                        "greedy_answer": result.greedy_answer,
                        "final_answer": result.final_answer,
                        "final_answer_source": str(result.final_answer_source),
                        JUDGE_ANSWER_FIELD: judge_answer,
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
