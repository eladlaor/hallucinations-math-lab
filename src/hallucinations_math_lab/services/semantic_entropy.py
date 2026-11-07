"""Semantic Entropy (Farquhar et al., Nature 2024).

Algorithm:
  1. Sample N generations at T>0 (handled by the vLLM client; this module receives them).
  2. Greedy-cluster the N samples by bidirectional NLI entailment against each cluster's
     representative (the first member). O(N * |clusters|) NLI calls instead of O(N^2).
     Each answer is prefixed with the question before NLI, as in the reference code.
  3. Turn the clusters into a distribution p(c), one of two ways
     (`SemanticEntropyEstimator`):
       - likelihood-weighted (default, the paper's main estimator): each sample
         weighs exp(mean token logprob), its length-normalized sequence
         likelihood; p(c) is the cluster's share of the total weight.
       - discrete: p(c) = |c| / N (the paper's black-box variant).
     SE = -Σ p(c) log p(c), in nats.

High SE = answers disagree in MEANING -> likely hallucination.
Low SE  = answers agree in meaning -> likely confident.
"""

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from hallucinations_math_lab.constants import (
    LOG_EXTRA_KEY,
    SE_NLI_INPUT_TEMPLATE,
    DetectorCategory,
    DetectorName,
    SemanticEntropyEstimator,
)
from hallucinations_math_lab.custom_types.domain_types import Cluster, SampleSet
from hallucinations_math_lab.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorMetadata,
    DetectorPrerequisiteError,
    DetectorScore,
)
from hallucinations_math_lab.services.nli import EntailmentScorer, get_nli_scorer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticEntropyResult:
    semantic_entropy: float
    clusters: list[Cluster]


@dataclass(frozen=True)
class SemanticEntropyByEstimator:
    """SE under several estimators, all computed from the same clustering."""

    by_estimator: dict[SemanticEntropyEstimator, float]
    clusters: list[Cluster]


def cluster_by_bidirectional_entailment(
    answers: list[str],
    scorer: EntailmentScorer | None = None,
    question: str | None = None,
) -> list[Cluster]:
    """Greedy single-pass clustering by bidirectional NLI entailment.

    Each new answer is compared to the representative of each existing cluster. If it
    bidirectionally entails any representative, it joins that cluster; otherwise it
    starts a new one. When `question` is given, NLI judges question-prefixed answers;
    `representative_text` is always the bare answer.
    """
    if not answers:
        raise ValueError("Cannot cluster empty answer list.")

    nli = scorer or get_nli_scorer()
    nli_texts = (
        answers
        if question is None
        else [SE_NLI_INPUT_TEMPLATE.format(question=question, answer=a) for a in answers]
    )
    clusters: list[Cluster] = []
    rep_nli_texts: list[str] = []

    for idx, ans in enumerate(answers):
        placed = False
        for cl, rep_nli_text in zip(clusters, rep_nli_texts, strict=True):
            try:
                if nli.bidirectional_entails(rep_nli_text, nli_texts[idx]):
                    cl.member_indices.append(idx)
                    placed = True
                    break
            except RuntimeError as e:
                logger.error(
                    "NLI entailment check failed; aborting clustering.",
                    extra={LOG_EXTRA_KEY: {"error": str(e), "answer_index": idx}},
                )
                raise
        if not placed:
            clusters.append(Cluster(member_indices=[idx], representative_text=ans))
            rep_nli_texts.append(nli_texts[idx])

    return clusters


def _sample_log_weights(sample_set: SampleSet) -> np.ndarray:
    """Length-normalized sequence log-likelihood per sample: its mean token logprob.

    Fails fast on a sample without logprobs or with a non-finite logprob;
    silently dropping or zeroing either would bias the cluster distribution.
    """
    weights: list[float] = []
    for idx, sample in enumerate(sample_set.samples):
        if not sample.token_logprobs:
            raise ValueError(
                f"Likelihood-weighted SE needs token logprobs, but sample {idx} has none "
                f"(prompt={sample_set.prompt[:80]!r}). Use the discrete estimator for "
                "providers without logprobs."
            )
        logprobs = [t.logprob for t in sample.token_logprobs]
        if not all(math.isfinite(lp) for lp in logprobs):
            raise ValueError(
                f"Sample {idx} has a non-finite token logprob (prompt={sample_set.prompt[:80]!r})."
            )
        weights.append(sum(logprobs) / len(logprobs))
    return np.asarray(weights, dtype=np.float64)


def _cluster_probs(
    clusters: list[Cluster], sample_set: SampleSet, estimator: SemanticEntropyEstimator
) -> np.ndarray:
    n_total = len(sample_set.samples)
    if estimator is SemanticEntropyEstimator.DISCRETE:
        return np.array([cl.size / n_total for cl in clusters], dtype=np.float64)
    log_w = _sample_log_weights(sample_set)
    # Normalize in log space (max-shift) so tiny likelihoods do not underflow.
    w = np.exp(log_w - log_w.max())
    cluster_w = np.array([w[cl.member_indices].sum() for cl in clusters], dtype=np.float64)
    return cluster_w / cluster_w.sum()


def _entropy_nats(probs: np.ndarray) -> float:
    # Natural-log entropy (nats), as in the paper. Every p(c) > 0, so no epsilon
    # is needed, and a single cluster scores exactly 0.
    return float(-np.sum(probs * np.log(probs))) if len(probs) > 1 else 0.0


def compute_semantic_entropies(
    sample_set: SampleSet,
    estimators: tuple[SemanticEntropyEstimator, ...],
    scorer: EntailmentScorer | None = None,
) -> SemanticEntropyByEstimator:
    """Cluster the samples once and score every requested estimator on those clusters.

    The NLI clustering is the expensive part, so asking for several estimators costs
    no extra NLI calls. Raises if the sample set or `estimators` is empty, or if the
    likelihood-weighted estimator is requested for samples without finite logprobs
    (checked before any NLI call).
    """
    if not sample_set.samples:
        raise ValueError(
            f"Cannot compute SE: empty sample set for prompt={sample_set.prompt[:80]!r}"
        )
    if not estimators:
        raise ValueError(
            f"Cannot compute SE: no estimators requested for prompt={sample_set.prompt[:80]!r}"
        )
    if SemanticEntropyEstimator.LIKELIHOOD_WEIGHTED in estimators:
        _sample_log_weights(sample_set)  # validate before paying for NLI calls

    answers = [s.text.strip() for s in sample_set.samples]
    clusters = cluster_by_bidirectional_entailment(
        answers, scorer=scorer, question=sample_set.prompt
    )
    by_estimator = {
        estimator: _entropy_nats(_cluster_probs(clusters, sample_set, estimator))
        for estimator in estimators
    }

    logger.info(
        "Computed semantic entropy.",
        extra={
            LOG_EXTRA_KEY: {
                "se_nats_by_estimator": {str(e): round(se, 4) for e, se in by_estimator.items()},
                "n_clusters": len(clusters),
                "n_samples": len(answers),
                "prompt": sample_set.prompt[:80],
            }
        },
    )

    return SemanticEntropyByEstimator(by_estimator=by_estimator, clusters=clusters)


def compute_semantic_entropy(
    sample_set: SampleSet,
    scorer: EntailmentScorer | None = None,
    estimator: SemanticEntropyEstimator = SemanticEntropyEstimator.LIKELIHOOD_WEIGHTED,
) -> SemanticEntropyResult:
    """Compute SE for a sample set with one estimator. Raises if the sample set is
    empty, or if the likelihood-weighted estimator gets samples without finite logprobs."""
    result = compute_semantic_entropies(sample_set, (estimator,), scorer=scorer)
    return SemanticEntropyResult(
        semantic_entropy=result.by_estimator[estimator], clusters=result.clusters
    )


class SemanticEntropyDetector(Detector):
    """Likelihood-weighted Semantic Entropy, Farquhar et al.'s main estimator."""

    name = DetectorName.SEMANTIC_ENTROPY
    estimator = SemanticEntropyEstimator.LIKELIHOOD_WEIGHTED
    requires_samples = True
    requires_hidden_state = False
    metadata = DetectorMetadata(
        signal="Shannon entropy over the meaning clusters of N samples (grouped by bidirectional NLI entailment), each cluster weighted by its samples' length-normalized sequence likelihoods",
        assumption="when the model is hallucinating, independent samples diverge in meaning even when each individually sounds fluent",
        failure_modes=(
            "NLI noise on long-form answers — clustering becomes unreliable past a few sentences",
            "consistent paraphrase variation inflates SE even when meaning is stable",
            "fixed N=10 caps the resolution; small N can miss low-probability divergent meanings",
        ),
        category=DetectorCategory.UNCERTAINTY,
    )

    def __init__(self, scorer: EntailmentScorer | None = None) -> None:
        self._scorer = scorer

    def score(self, ctx: DetectionContext) -> DetectorScore:
        if ctx.samples is None:
            raise DetectorPrerequisiteError(
                "SemanticEntropyDetector requires samples; DetectionContext.samples is None."
            )
        # The dominant cost is the O(N * |clusters|) NLI calls, paid in
        # wall-clock GPU time on the NLI model rather than $/call against an
        # external API. cost_usd is 0 (local GPU); latency_ms captures the
        # wall-clock cost so the Pareto frontier still ranks SE realistically.
        start = time.perf_counter()
        result = compute_semantic_entropy(
            ctx.samples, scorer=self._scorer, estimator=self.estimator
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DetectorScore(
            name=self.name,
            score=result.semantic_entropy,
            cost_usd=0.0,
            latency_ms=elapsed_ms,
            clusters=result.clusters,
        )


class DiscreteSemanticEntropyDetector(SemanticEntropyDetector):
    """Discrete Semantic Entropy: p(c) = |c| / N, no token logprobs needed.

    The paper's variant for black-box models. Enabling it next to
    `SemanticEntropyDetector` clusters the samples a second time, doubling
    the NLI cost.
    """

    name = DetectorName.DISCRETE_SEMANTIC_ENTROPY
    estimator = SemanticEntropyEstimator.DISCRETE
    metadata = DetectorMetadata(
        signal="Shannon entropy over the cluster-size distribution (|c|/N) of N samples grouped by bidirectional NLI entailment",
        assumption=SemanticEntropyDetector.metadata.assumption,
        failure_modes=SemanticEntropyDetector.metadata.failure_modes,
        category=DetectorCategory.UNCERTAINTY,
    )
