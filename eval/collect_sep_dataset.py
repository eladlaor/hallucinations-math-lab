"""Collect the SEP dataset: SE labels, answers and every layer's hidden states.

For each calibration question:
  1. Sample N answers and cluster them once by bidirectional NLI. Both SE
     estimators come from that one clustering, so storing the
     likelihood-weighted value next to the discrete one costs no extra NLI
     calls. The probe trains on discrete SE (SEP_CLASSIFIER.md, "Which SE goes
     into the label?"); the likelihood-weighted value is kept because it is
     the SE detector's default score.
  2. Greedy-decode one answer and label it correct when a ground-truth alias
     appears in it (eval/benchmark_results.py's rule).
  3. Read every layer's hidden state at TBG (the last prompt token) and at SLT
     (teacher-forcing the greedy answer).

The output is a dataset directory (eval/sep_dataset_io.py), the probe fit's
only input: rows.parquet, hidden_states.safetensors, collection_config.json
and training_log.jsonl, the last appended one line per question as it runs.
The other three are written at the end, or, when a question fails or the run
is interrupted, for the questions finished before it. collection_config.json
names the calibration set by file name and SHA-256 and records how many
questions it holds, so such a partial dataset is recognisable, and load_sep_dataset refuses it unless asked not to.

Collection needs SLM_PROVIDER=hf_transformers: the layer sweep needs every
layer, and the vLLM embed runner pools only the last one. It also needs a GPU
for any real backbone.

Usage:
    uv run python -m eval.collect_sep_dataset [--calibration-set FILE] [--output-dir DIR]

Without `--calibration-set`, the default 500-question TriviaQA calibration
slice is used, downloaded on first use (eval/datasets/default_question_sets.py).
Without `--output-dir`, the dataset goes to data/sep_datasets/<model slug>/.
The output directory must be empty or absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import numpy as np
import torch

from demo.model_registry import get_model_entry
from demo.slm_provider import SamplingTruncation, build_default_provider
from eval.benchmark_results import ALIASES_FIELD, ID_FIELD, QUESTION_FIELD, is_correct
from eval.datasets.default_question_sets import CALIBRATION_SLICE, ensure_question_set
from eval.sep_dataset_io import (
    SEPDataset,
    SEPDatasetRow,
    TrainingLogEntry,
    append_training_log,
    prepare_dataset_dir,
    write_sep_dataset,
)
from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import (
    HUB_COMMIT_SHA_PATTERN,
    LOG_EXTRA_KEY,
    SEP_DATASET_DIR,
    SemanticEntropyEstimator,
    TokenPosition,
)
from hallucinations_math_lab.custom_types.domain_types import Cluster, SampleSet
from hallucinations_math_lab.custom_types.sep_probe_types import (
    SEPBackbone,
    SEPCollectionConfig,
    SEPCollectionLabels,
    SEPPromptSpec,
)
from hallucinations_math_lab.services.nli import EntailmentScorer
from hallucinations_math_lab.services.sampling_provider import SamplingProvider
from hallucinations_math_lab.services.semantic_entropy import compute_semantic_entropies
from hallucinations_math_lab.services.semantic_entropy_probe import model_slug

logger = logging.getLogger("collect_sep_dataset")

LABEL_ESTIMATORS = (
    SemanticEntropyEstimator.DISCRETE,
    SemanticEntropyEstimator.LIKELIHOOD_WEIGHTED,
)
_UNASSIGNED_CLUSTER_ID = -1
_PROMPT_PREVIEW_CHARS: Final[int] = 80


@dataclass(frozen=True)
class SELabels:
    """One question's SE fields in the dataset row."""

    se_discrete: float
    se_likelihood_weighted: float
    cluster_ids: list[int]  # cluster index per sample, in sample order
    n_clusters: int


@dataclass(frozen=True)
class CalibrationQuestion:
    """One row of a calibration JSONL (load_triviaqa.py output)."""

    id: str
    question: str
    ground_truth_aliases: list[str]


@dataclass(frozen=True)
class CalibrationSet:
    """A calibration JSONL's questions, with the file name and the SHA-256 of
    its bytes that collection_config.json records."""

    name: str
    sha256: str
    questions: list[CalibrationQuestion]


@runtime_checkable
class AllLayerHiddenStateProvider(SamplingProvider, Protocol):
    """A sampling provider that can also read every layer at TBG or SLT and
    name its chat template and sampling truncation. HFTransformersProvider
    satisfies it; the vLLM provider does not."""

    @property
    def chat_template_sha256(self) -> str: ...

    @property
    def sampling_truncation(self) -> SamplingTruncation: ...

    def hidden_states_all_layers(
        self, prompt: str, position: TokenPosition, answer: str | None = None
    ) -> torch.Tensor: ...


def label_sample_set(sample_set: SampleSet, scorer: EntailmentScorer | None = None) -> SELabels:
    """Discrete and likelihood-weighted SE, plus cluster ids, from one clustering.

    Raises before any NLI call if a sample lacks finite token logprobs.
    """
    result = compute_semantic_entropies(sample_set, LABEL_ESTIMATORS, scorer=scorer)
    return SELabels(
        se_discrete=result.by_estimator[SemanticEntropyEstimator.DISCRETE],
        se_likelihood_weighted=result.by_estimator[SemanticEntropyEstimator.LIKELIHOOD_WEIGHTED],
        cluster_ids=cluster_ids_per_sample(result.clusters, sample_set),
        n_clusters=len(result.clusters),
    )


def cluster_ids_per_sample(clusters: list[Cluster], sample_set: SampleSet) -> list[int]:
    """Cluster index of each sample, in sample order.

    Raises if the clusters do not partition the samples: a sample in no cluster or
    in two would otherwise be stored with a wrong cluster id.
    """
    cluster_ids = [_UNASSIGNED_CLUSTER_ID] * len(sample_set.samples)
    for cluster_id, cluster in enumerate(clusters):
        for member in cluster.member_indices:
            if cluster_ids[member] != _UNASSIGNED_CLUSTER_ID:
                raise ValueError(
                    f"Sample {member} is in clusters {cluster_ids[member]} and {cluster_id} "
                    f"(prompt={sample_set.prompt[:_PROMPT_PREVIEW_CHARS]!r})."
                )
            cluster_ids[member] = cluster_id
    missing = [i for i, cid in enumerate(cluster_ids) if cid == _UNASSIGNED_CLUSTER_ID]
    if missing:
        raise ValueError(
            f"Samples {missing} are in no cluster "
            f"(prompt={sample_set.prompt[:_PROMPT_PREVIEW_CHARS]!r})."
        )
    return cluster_ids


def read_calibration_set(path: Path) -> CalibrationSet:
    """Every question in a calibration JSONL, hashed from the same bytes it is
    parsed from; fails fast on a malformed row, a row without aliases (greedy
    correctness needs them) or a repeated id.

    Fields are type-checked, not coerced: a null question would otherwise be
    collected as the prompt "None"."""
    questions: list[CalibrationQuestion] = []
    try:
        content = path.read_bytes()
    except OSError as e:
        raise RuntimeError(f"Cannot read calibration set {path}: {e!s}") from e
    try:
        lines = content.decode().splitlines()
    except UnicodeDecodeError as e:
        raise ValueError(f"Calibration set {path} is not UTF-8: {e!s}") from e
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            questions.append(_parse_calibration_row(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"Calibration set {path} line {line_no} needs {ID_FIELD}, {QUESTION_FIELD} and "
                f"{ALIASES_FIELD} (build it with eval.datasets.load_triviaqa): {e!r}"
            ) from e
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Calibration set {path} has duplicate question ids: {duplicates}.")
    return CalibrationSet(
        name=path.name, sha256=hashlib.sha256(content).hexdigest(), questions=questions
    )


def _parse_calibration_row(obj: object) -> CalibrationQuestion:
    """One decoded JSONL line as a question; raises TypeError or ValueError on a
    wrong shape, type or value."""
    if not isinstance(obj, dict):
        raise TypeError(f"a row must be a JSON object, got {type(obj).__name__}")
    question_id, question, aliases = obj[ID_FIELD], obj[QUESTION_FIELD], obj[ALIASES_FIELD]
    # bool is an int subclass, and `true` is no question id.
    if isinstance(question_id, bool) or not isinstance(question_id, str | int):
        raise TypeError(f"{ID_FIELD} must be a string or an integer, got {question_id!r}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{QUESTION_FIELD} must be a non-empty string, got {question!r}")
    if (
        not isinstance(aliases, list)
        or not aliases
        or not all(isinstance(a, str) and a.strip() for a in aliases)
    ):
        raise ValueError(f"{ALIASES_FIELD} must be a non-empty list of strings, got {aliases!r}")
    return CalibrationQuestion(id=str(question_id), question=question, ground_truth_aliases=aliases)


def require_all_layer_provider(provider: object) -> AllLayerHiddenStateProvider:
    """The provider, narrowed; raises unless it can read every layer."""
    if not isinstance(provider, AllLayerHiddenStateProvider):
        raise TypeError(
            f"SEP dataset collection reads every layer's hidden state, which "
            f"{type(provider).__name__} cannot provide (the vLLM embed runner pools only the "
            "last layer). Collect with SLM_PROVIDER=hf_transformers."
        )
    return provider


def build_collection_config(
    provider: AllLayerHiddenStateProvider, calibration_set: CalibrationSet
) -> SEPCollectionConfig:
    """The calibration set's identity and size, and the backbone, prompt and
    labels blocks for this run.

    The backbone comes from the model registry, and the provider must report
    the registry's pinned revision and depth: a dataset is useless unless it
    names the exact checkpoint its hidden states came from."""
    entry = get_model_entry(provider.model_id)
    revision = provider.model_revision
    if not re.fullmatch(HUB_COMMIT_SHA_PATTERN, revision) or revision != entry.revision:
        raise ValueError(
            f"{provider.model_id} reports revision {revision!r}, but the model registry pins "
            f"{entry.revision}. SEP collection needs the exact pinned commit."
        )
    if provider.num_hidden_layers != entry.num_hidden_layers:
        raise ValueError(
            f"{provider.model_id} reports {provider.num_hidden_layers} decoder layers, but the "
            f"model registry has {entry.num_hidden_layers}."
        )
    settings = get_settings()
    truncation = provider.sampling_truncation
    return SEPCollectionConfig(
        calibration_set=calibration_set.name,
        calibration_set_sha256=calibration_set.sha256,
        n_questions_requested=len(calibration_set.questions),
        backbone=SEPBackbone(
            model_id=entry.model_id,
            revision=entry.revision,
            num_hidden_layers=entry.num_hidden_layers,
            hidden_size=entry.hidden_size,
            dtype=entry.dtype,
        ),
        prompt=SEPPromptSpec(
            chat_template_sha256=provider.chat_template_sha256,
            answer_instruction=settings.answer_instruction,
        ),
        labels=SEPCollectionLabels(
            sample_n=settings.sample_n,
            temperature=settings.sample_temperature,
            top_p=truncation.top_p,
            top_k=truncation.top_k,
            max_tokens=settings.max_tokens,
            nli_model_id=settings.nli_model_id,
            nli_revision=settings.nli_revision,
        ),
    )


def _hidden_states_array(
    states: torch.Tensor | None, position: TokenPosition, backbone: SEPBackbone
) -> np.ndarray:
    """One position's [L+1, hidden_size] states as float32 numpy; raises on None
    or a shape that disagrees with the backbone."""
    if states is None:
        raise RuntimeError(f"The provider returned no hidden states at {position}.")
    expected = (backbone.num_hidden_layers + 1, backbone.hidden_size)
    if tuple(states.shape) != expected:
        raise RuntimeError(
            f"Hidden states at {position} have shape {tuple(states.shape)}; expected {expected} "
            "(L+1, hidden_size)."
        )
    return states.detach().to(device="cpu", dtype=torch.float32).numpy()


def collect_question(
    provider: AllLayerHiddenStateProvider,
    question: CalibrationQuestion,
    backbone: SEPBackbone,
    scorer: EntailmentScorer | None,
) -> tuple[SEPDatasetRow, dict[TokenPosition, np.ndarray], TrainingLogEntry]:
    """One question's row, its hidden states per position, and its log line."""
    started = time.perf_counter()
    sample_set = provider.sample(question.question)
    labels = label_sample_set(sample_set, scorer=scorer)
    greedy_answer = provider.greedy(question.question)
    greedy_correct = is_correct(greedy_answer, question.ground_truth_aliases)
    hidden_states = {
        TokenPosition.TBG: _hidden_states_array(
            provider.hidden_states_all_layers(question.question, TokenPosition.TBG),
            TokenPosition.TBG,
            backbone,
        ),
        TokenPosition.SLT: _hidden_states_array(
            provider.hidden_states_all_layers(
                question.question, TokenPosition.SLT, answer=greedy_answer
            ),
            TokenPosition.SLT,
            backbone,
        ),
    }
    sampled_answers = [sample.text for sample in sample_set.samples]
    row = SEPDatasetRow(
        id=question.id,
        question=question.question,
        ground_truth_aliases=question.ground_truth_aliases,
        sampled_answers=sampled_answers,
        cluster_ids=labels.cluster_ids,
        se_discrete=labels.se_discrete,
        se_likelihood_weighted=labels.se_likelihood_weighted,
        n_clusters=labels.n_clusters,
        greedy_answer=greedy_answer,
        greedy_correct=greedy_correct,
    )
    log_entry = TrainingLogEntry(
        id=question.id,
        se_discrete=labels.se_discrete,
        se_likelihood_weighted=labels.se_likelihood_weighted,
        n_clusters=labels.n_clusters,
        sampled_answer_word_counts=[len(answer.split()) for answer in sampled_answers],
        greedy_answer_word_count=len(greedy_answer.split()),
        greedy_correct=greedy_correct,
        elapsed_s=time.perf_counter() - started,
    )
    return row, hidden_states, log_entry


def collect_sep_dataset(
    provider: object,
    calibration_set: CalibrationSet,
    output_dir: Path,
    scorer: EntailmentScorer | None = None,
) -> SEPDataset:
    """Collect every question and write the dataset directory.

    The provider, the output directory and the pinned revision are checked
    before the first sample; the directory check comes before anything that
    loads the model. A failure anywhere in a question, from sampling to its
    training-log line, stops the run, naming that question, after writing the
    dataset of the questions finished before it. An interrupt (Ctrl-C) writes
    the same partial dataset and is re-raised as is.

    `scorer` is a test seam for a fake NLI. The written config always records
    the configured NLI_MODEL_ID and NLI_REVISION, which the default scorer
    loads, so a real run passes none."""
    collector = require_all_layer_provider(provider)
    questions = calibration_set.questions
    if not questions:
        raise ValueError(f"SEP collection got no questions from {calibration_set.name}.")
    prepare_dataset_dir(output_dir)
    config = build_collection_config(collector, calibration_set)

    backbone = config.backbone
    shape = (len(questions), backbone.num_hidden_layers + 1, backbone.hidden_size)
    hidden_states = {position: np.empty(shape, dtype=np.float32) for position in TokenPosition}
    rows: list[SEPDatasetRow] = []
    for index, question in enumerate(questions):
        ctx = {"index": index, "n_questions": len(questions), "question_id": question.id}
        try:
            row, states, log_entry = collect_question(collector, question, backbone, scorer)
            for position, array in states.items():
                hidden_states[position][index] = array
            append_training_log(output_dir, log_entry)
            # Last, so `rows` only ever holds questions whose states and log line are stored.
            rows.append(row)
        except BaseException as e:
            # BaseException, so an interrupted hours-long run keeps its finished rows too.
            logger.error(
                "SEP collection stopped.",
                extra={LOG_EXTRA_KEY: {**ctx, "error": repr(e)}},
            )
            finished = SEPDataset(
                config=config,
                rows=rows,
                hidden_states={p: done[: len(rows)] for p, done in hidden_states.items()},
            )
            kept = _write_finished_questions(output_dir, finished, question, e)
            if not isinstance(e, Exception):
                raise
            raise RuntimeError(
                f"SEP collection failed at question {question.id!r} ({index + 1}/"
                f"{len(questions)}, prompt={question.question[:_PROMPT_PREVIEW_CHARS]!r}); "
                f"{kept}: {e!s}"
            ) from e
        logger.info(
            "Collected SEP dataset row.",
            extra={
                LOG_EXTRA_KEY: {
                    **ctx,
                    "se_discrete": round(row.se_discrete, 4),
                    "greedy_correct": row.greedy_correct,
                    "elapsed_s": round(log_entry.elapsed_s, 2),
                }
            },
        )

    dataset = SEPDataset(config=config, rows=rows, hidden_states=hidden_states)
    write_sep_dataset(output_dir, dataset)
    logger.info(
        "Wrote SEP dataset.",
        extra={
            LOG_EXTRA_KEY: {
                "output_dir": str(output_dir),
                "model_id": backbone.model_id,
                "revision": backbone.revision,
                "n_rows": len(rows),
            }
        },
    )
    return dataset


def _write_finished_questions(
    output_dir: Path,
    finished: SEPDataset,
    failed_question: CalibrationQuestion,
    error: BaseException,
) -> str:
    """Write the questions finished before `failed_question`, if any, and say
    what was kept. A failed write raises naming both the question and `error`."""
    if not finished.rows:
        return "no dataset was written"
    n_finished = len(finished.rows)
    try:
        write_sep_dataset(output_dir, finished)
    except Exception as write_error:
        raise RuntimeError(
            f"SEP collection stopped at question {failed_question.id!r} ({error!r}), and "
            f"writing the {n_finished} questions before it to {output_dir} also failed: "
            f"{write_error!s}"
        ) from write_error
    return f"the {n_finished} questions before it were written to {output_dir}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect the SEP calibration dataset.")
    parser.add_argument(
        "--calibration-set",
        type=Path,
        default=None,
        help=f"Question JSONL; default: {CALIBRATION_SLICE.path}, downloaded on first use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Empty or absent directory; default: {SEP_DATASET_DIR}/<model slug>/.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    calibration_set = read_calibration_set(
        args.calibration_set or ensure_question_set(CALIBRATION_SLICE)
    )
    provider = build_default_provider()
    output_dir = args.output_dir or Path(SEP_DATASET_DIR) / model_slug(provider.model_id)
    collect_sep_dataset(provider, calibration_set, output_dir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.exit(main())
