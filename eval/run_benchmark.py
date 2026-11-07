"""Run the configured detectors over a question set and write one JSONL row per question.

Input: JSONL with a `question` field per line (eval/datasets/load_triviaqa.py
output, or eval/datasets/adversarial.jsonl). Without `--in`, the default
200-question TriviaQA slice is used, downloaded on first use
(eval/datasets/default_question_sets.py). Ground-truth fields
(`ground_truth_aliases`, `ground_truth_value`, `expected_behavior`) are copied
through untouched so compute_auroc.py can label each row.

Detectors and the escalation rule come from settings (ENABLED_DETECTORS,
ESCALATION_DETECTOR, and SE_THRESHOLD or, for SEP, SEP_THRESHOLD); the SLM
comes from SLM_PROVIDER / VLLM_MODEL_ID. Every row records all detector
scores under `detector_scores`, so adding a detector needs no change here or
in compute_auroc.py.

With a judge, every row records the judge's answer as `judge_answer`, not
only the escalated rows, so escalation_uplift.py can compare the detector's
escalations with random ones at the same rate. This costs one judge call per
question.

Every row also records the run's `model_id`, `sample_n`,
`sample_temperature`, `max_tokens`, `answer_instruction` (the brief-answer
prompt) and `sep_probe`: the SEP probe's reference, Hub commit and file
hashes, or null without SEP. The scoring scripts refuse to mix rows from
different runs or to compare runs that differ in them.

When a Semantic Entropy Probe is among the detectors, the run fails before
scoring anything if a question was in the probe's training set (see
eval/sep_probe_provenance.py).

Two modes:
  - one model (default): VLLM_MODEL_ID, `--out` is a results file.
  - `--sweep`: every id in TARGET_MODEL_IDS, one after another, `--out` is a
    directory that receives `<model slug>.jsonl` per model. Each model is
    built, run and released before the next loads. Every model and its SEP
    probe are checked before the first question is scored: the probe must
    exist, never have seen these questions, and match the model's registry
    entry (the chat template is checked when each model is built).

Downstream: compute_auroc.py, escalation_uplift.py, adversarial_separation.py,
compare_models.py (for a sweep).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from demo.model_registry import get_model_entry
from eval.benchmark_results import (
    ANSWER_INSTRUCTION_FIELD,
    ID_FIELD,
    JUDGE_ANSWER_FIELD,
    MAX_TOKENS_FIELD,
    MODEL_ID_FIELD,
    QUESTION_FIELD,
    SAMPLE_N_FIELD,
    SAMPLE_TEMPERATURE_FIELD,
    SEP_PROBE_FIELD,
    RunConfig,
)
from eval.datasets.default_question_sets import (
    BENCHMARK_SLICE,
    QuestionLoader,
    QuestionSlice,
    ensure_question_set,
)
from eval.sep_probe_provenance import assert_disjoint, probe_run_record
from hallucinations_math_lab import ResponseEvaluator
from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import LOG_EXTRA_KEY, AnswerSource, DetectorName
from hallucinations_math_lab.custom_types.domain_types import DetectionResult
from hallucinations_math_lab.services.semantic_entropy_probe import (
    SemanticEntropyProbeDetector,
    SEPProbeMismatchError,
    default_probe_dir,
    model_slug,
    resolve_probe_source,
)

RESULTS_SUFFIX = ".jsonl"

EvaluatorFactory = Callable[[str], ResponseEvaluator]

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


def _sep_probe(evaluator: ResponseEvaluator) -> SemanticEntropyProbeDetector | None:
    """The evaluator's SEP detector, if it has one."""
    probes = [d for d in evaluator.detectors if isinstance(d, SemanticEntropyProbeDetector)]
    if len(probes) > 1:
        raise ValueError(f"The evaluator has {len(probes)} SEP detectors; a run records one probe.")
    return probes[0] if probes else None


def _question_ids(questions: list[dict[str, Any]]) -> list[str]:
    """Every question's id, for the SEP leakage guard; a question without one cannot be
    shown absent from the probe's training set, so it is refused."""
    ids: list[str] = []
    for row_no, question in enumerate(questions, start=1):
        question_id = question.get(ID_FIELD)
        if question_id is None:
            raise ValueError(
                f"Question {row_no} ({question.get(QUESTION_FIELD)!r}) has no {ID_FIELD!r}; "
                "with a SEP probe every question needs one, to check it was not used to train "
                "the probe. Give every row an id, or drop semantic_entropy_probe from "
                "ENABLED_DETECTORS."
            )
        ids.append(str(question_id))
    return ids


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


def _judge_or_none(judge: bool):
    if not judge:
        return None
    from hallucinations_math_lab.judges import GeminiJudge

    return GeminiJudge()


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
        evaluator = ResponseEvaluator(judge=_judge_or_none(judge))

    questions = _read_questions(in_path, limit)
    probe = _sep_probe(evaluator)
    if probe is not None:
        assert_disjoint(probe, _question_ids(questions))
    settings = get_settings()
    run_config: RunConfig = {
        MODEL_ID_FIELD: evaluator.slm.model_id,
        SAMPLE_N_FIELD: settings.sample_n,
        SAMPLE_TEMPERATURE_FIELD: settings.sample_temperature,
        MAX_TOKENS_FIELD: settings.max_tokens,
        ANSWER_INSTRUCTION_FIELD: settings.answer_instruction,
        SEP_PROBE_FIELD: None if probe is None else probe_run_record(probe),
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


def _sweep_train_hint(model_id: str) -> str:
    """How to give `model_id` its own probe in a sweep, which refuses SEP_PROBE_PATH."""
    return (
        f"Collect a dataset with `VLLM_MODEL_ID={model_id} uv run python -m "
        "eval.collect_sep_dataset`, then train with `uv run python -m "
        "eval.train_sep_probe --dataset DIR`, or drop semantic_entropy_probe from "
        "ENABLED_DETECTORS."
    )


def _preflight_sweep_probes(model_ids: list[str], questions: list[dict[str, Any]]) -> None:
    """With SEP enabled, every model needs its own probe that fits its registry entry
    and never saw these questions. Loading each probe here also validates its files."""
    settings = get_settings()
    if DetectorName.SEMANTIC_ENTROPY_PROBE not in settings.enabled_detectors:
        return
    if settings.sep_probe_path:
        raise ValueError(
            f"SEP_PROBE_PATH={settings.sep_probe_path!r} names one probe, but a sweep runs "
            f"{len(model_ids)} models and each needs its own. Unset SEP_PROBE_PATH so every "
            "model uses data/probes/<model slug>/."
        )
    ids = _question_ids(questions)
    for model_id in model_ids:
        try:
            # Refuses a legacy .pt probe (ObsoleteSEPProbeError) as it is.
            source = resolve_probe_source("", model_id)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No SEP probe for {model_id} at {default_probe_dir(model_id)}. "
                f"{_sweep_train_hint(model_id)}"
            ) from e
        entry = get_model_entry(model_id)
        probe = SemanticEntropyProbeDetector(source)
        try:
            probe.check_backbone(
                model_id=model_id,
                revision=entry.revision,
                num_hidden_layers=entry.num_hidden_layers,
                hidden_size=entry.hidden_size,
                answer_instruction=settings.answer_instruction,
            )
        except SEPProbeMismatchError as e:
            # The generic hint suggests SEP_PROBE_PATH, which a sweep refuses.
            raise SEPProbeMismatchError(
                e.probe_dir,
                e.mismatches,
                hint=f"Move {default_probe_dir(model_id)} aside and train {model_id}'s probe "
                "in its place. "
                f"{_sweep_train_hint(model_id)}",
            ) from e
        assert_disjoint(probe, ids)


def run_sweep(
    in_path: Path,
    out_dir: Path,
    judge: bool = True,
    limit: int | None = None,
    evaluator_factory: EvaluatorFactory | None = None,
) -> list[Path]:
    """Run the benchmark once per model in TARGET_MODEL_IDS; return the results files.

    `evaluator_factory(model_id)` builds the evaluator for one model; the
    default wraps `build_default_provider(model_id=...)`. Only one evaluator
    is alive at a time, so a local sweep holds one model's weights at once.
    """
    from demo.slm_provider import build_default_provider, sweep_model_ids

    model_ids = sweep_model_ids()
    _preflight_sweep_probes(model_ids, _read_questions(in_path, limit))
    if evaluator_factory is None:
        judge_impl = _judge_or_none(judge)

        def evaluator_factory(model_id: str) -> ResponseEvaluator:
            return ResponseEvaluator(
                slm=build_default_provider(model_id=model_id), judge=judge_impl
            )

    out_paths: list[Path] = []
    for position, model_id in enumerate(model_ids, start=1):
        out_path = out_dir / f"{model_slug(model_id)}{RESULTS_SUFFIX}"
        logger.info(
            "Sweep: starting model.",
            extra={
                LOG_EXTRA_KEY: {
                    "model_id": model_id,
                    "position": position,
                    "n_models": len(model_ids),
                    "out_path": str(out_path),
                }
            },
        )
        evaluator = evaluator_factory(model_id)
        run(in_path, out_path, judge=judge, limit=limit, evaluator=evaluator)
        out_paths.append(out_path)
        del evaluator
        gc.collect()
    return out_paths


def resolve_questions_path(
    in_path: Path | None,
    loader: QuestionLoader | None = None,
    slice_: QuestionSlice = BENCHMARK_SLICE,
) -> Path:
    """The given question file, or the default benchmark slice (built on first use)."""
    return in_path if in_path is not None else ensure_question_set(slice_, loader=loader)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the detectors over a question set.")
    p.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=None,
        help=f"Question JSONL; default: {BENCHMARK_SLICE.path}, downloaded on first use.",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        required=True,
        help="Results file; with --sweep, a directory for one file per model.",
    )
    p.add_argument("--no-judge", action="store_true", help="Skip Gemini escalation.")
    p.add_argument("--limit", type=int, default=None, help="Stop after this many rows.")
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run every model in TARGET_MODEL_IDS instead of only VLLM_MODEL_ID.",
    )
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    in_path = resolve_questions_path(args.in_path)
    if args.sweep:
        run_sweep(in_path, args.out_path, judge=not args.no_judge, limit=args.limit)
    else:
        run(in_path, args.out_path, judge=not args.no_judge, limit=args.limit)
