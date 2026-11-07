"""Re-score a saved benchmark run without the model.

run_benchmark.py saves every sampled answer with its token logprobs. This
script re-runs the sample-based detectors (PE, SE, discrete SE) on those saved
answers and re-applies the escalation rule, so you can try a new threshold, a
new meaning check or a new detector on a frozen run: same answers, no GPU, no
model call. It writes a new results file that compute_auroc.py,
calibrate_threshold.py, escalation_uplift.py and compare_models.py read like
any other run.

What carries over unchanged: the question, the answer key, the run's model and
sampling settings, the greedy answer, the samples and the judge's recorded
answer. What is recomputed: `detector_scores`, `n_clusters`, `escalation`,
`escalated`, and from those `final_answer` / `final_answer_source` (an
escalated row takes the judge answer the run recorded, if any).

Refused, rather than scored wrongly:
  - SEP, which reads the model's hidden state and so needs the running model;
  - files written before runs saved `sample_token_logprobs`.

Detectors and the escalation rule come from settings (ENABLED_DETECTORS,
ESCALATION_DETECTOR, SE_THRESHOLD), as for run_benchmark.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from eval.benchmark_results import (
    DETECTOR_SCORES_FIELD,
    ESCALATED_FIELD,
    ESCALATION_FIELD,
    FINAL_ANSWER_FIELD,
    FINAL_ANSWER_SOURCE_FIELD,
    GREEDY_ANSWER_FIELD,
    ID_FIELD,
    JUDGE_ANSWER_FIELD,
    N_CLUSTERS_FIELD,
    QUESTION_FIELD,
    SAMPLE_TOKEN_LOGPROBS_FIELD,
    SAMPLES_FIELD,
    BenchmarkRow,
    load_rows,
    run_config,
)
from hallucinations_math_lab import evaluate_responses
from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import LOG_EXTRA_KEY, AnswerSource, DetectorName
from hallucinations_math_lab.custom_types.domain_types import Sample, SampleSet, TokenLogprob
from hallucinations_math_lab.services.detector_protocol import Detector

logger = logging.getLogger(__name__)


def _sample_set(row: BenchmarkRow, where: str) -> SampleSet:
    """The row's saved answers with their token logprobs, as the detectors take them."""
    if SAMPLE_TOKEN_LOGPROBS_FIELD not in row:
        raise RuntimeError(
            f"{where} has no {SAMPLE_TOKEN_LOGPROBS_FIELD!r}: it was written before runs saved "
            "token logprobs, so PE and SE cannot be recomputed from it. Re-run "
            "eval.run_benchmark to get a file that can be re-scored."
        )
    texts: list[str] = row[SAMPLES_FIELD]
    logprobs: list[list[dict]] = row[SAMPLE_TOKEN_LOGPROBS_FIELD]
    if len(logprobs) != len(texts):
        raise RuntimeError(
            f"{where} has {len(texts)} samples but {len(logprobs)} token-logprob lists; "
            "the file is corrupt."
        )
    return SampleSet(
        prompt=row[QUESTION_FIELD],
        samples=[
            Sample(text=text, token_logprobs=[TokenLogprob(**t) for t in tokens])
            for text, tokens in zip(texts, logprobs, strict=True)
        ],
    )


def _rescore_row(
    row: BenchmarkRow,
    where: str,
    detectors: list[Detector],
    escalation_detector: DetectorName,
    escalation_threshold: float,
) -> BenchmarkRow:
    result = evaluate_responses(
        prompt=row[QUESTION_FIELD],
        samples=_sample_set(row, where),
        detectors=detectors,
        escalation_detector=escalation_detector,
        escalation_threshold=escalation_threshold,
        greedy_answer=row[GREEDY_ANSWER_FIELD],
    )
    judge_answer: str | None = row.get(JUDGE_ANSWER_FIELD)
    use_judge = result.escalated and judge_answer is not None
    return {
        **row,
        DETECTOR_SCORES_FIELD: {str(name): score for name, score in result.detector_scores.items()},
        N_CLUSTERS_FIELD: len(result.clusters) if result.clusters else None,
        ESCALATION_FIELD: f"{escalation_detector} > {escalation_threshold}",
        ESCALATED_FIELD: result.escalated,
        FINAL_ANSWER_FIELD: judge_answer if use_judge else result.greedy_answer,
        FINAL_ANSWER_SOURCE_FIELD: str(
            AnswerSource.JUDGE_MODEL if use_judge else AnswerSource.SMALL_MODEL_GREEDY
        ),
    }


def rescore(
    in_path: Path,
    out_path: Path,
    *,
    detectors: list[Detector],
    escalation_detector: DetectorName,
    escalation_threshold: float,
) -> int:
    """Re-score every row of `in_path` into `out_path`; return the number of rows written."""
    if in_path.resolve() == out_path.resolve():
        raise ValueError(f"Refusing to overwrite the run being re-scored: {in_path}.")
    needs_model = [str(d.name) for d in detectors if d.requires_hidden_state]
    if needs_model:
        raise ValueError(
            f"{needs_model} read the model's hidden state, which a saved run does not keep, "
            "so they cannot be re-scored offline. Drop them, or re-run eval.run_benchmark."
        )

    rows = load_rows(in_path)
    run_config(rows, str(in_path))  # one run per file, as every scoring script requires
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f_out:
        for row_no, row in enumerate(rows, start=1):
            where = f"{in_path} row {row_no} (id={row.get(ID_FIELD)})"
            try:
                rescored = _rescore_row(
                    row, where, detectors, escalation_detector, escalation_threshold
                )
            except Exception as e:
                logger.error(
                    "Re-scoring failed; stopping.",
                    extra={LOG_EXTRA_KEY: {"where": where, "error": str(e)}},
                )
                raise
            f_out.write(json.dumps(rescored) + "\n")

    logger.info(
        "Re-scoring complete.",
        extra={
            LOG_EXTRA_KEY: {
                "in_path": str(in_path),
                "out_path": str(out_path),
                "rows_written": len(rows),
                "detectors": [str(d.name) for d in detectors],
                "escalation": f"{escalation_detector} > {escalation_threshold}",
            }
        },
    )
    return len(rows)


def _detectors_from_settings(enabled: list[DetectorName]) -> list[Detector]:
    """The sample-based detectors named in ENABLED_DETECTORS; SEP is refused."""
    from hallucinations_math_lab.services.predictive_entropy import PredictiveEntropyDetector
    from hallucinations_math_lab.services.semantic_entropy import (
        DiscreteSemanticEntropyDetector,
        SemanticEntropyDetector,
    )

    offline: dict[DetectorName, type[Detector]] = {
        DetectorName.PREDICTIVE_ENTROPY: PredictiveEntropyDetector,
        DetectorName.SEMANTIC_ENTROPY: SemanticEntropyDetector,
        DetectorName.DISCRETE_SEMANTIC_ENTROPY: DiscreteSemanticEntropyDetector,
    }
    unsupported = [str(name) for name in enabled if name not in offline]
    if unsupported:
        raise ValueError(
            f"ENABLED_DETECTORS includes {unsupported}, which cannot be re-scored from a saved "
            f"run (they need the running model). Re-scorable: {[str(n) for n in offline]}."
        )
    return [offline[name]() for name in enabled]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Re-run the sample-based detectors on a saved benchmark run, without the model."
    )
    p.add_argument("--in", dest="in_path", type=Path, required=True, help="A results file.")
    p.add_argument("--out", dest="out_path", type=Path, required=True, help="New results file.")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    settings = get_settings()
    rescore(
        args.in_path,
        args.out_path,
        detectors=_detectors_from_settings(settings.enabled_detectors),
        escalation_detector=settings.escalation_detector,
        escalation_threshold=settings.se_threshold,
    )
