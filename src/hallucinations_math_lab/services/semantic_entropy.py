"""Semantic Entropy (Farquhar et al., Nature 2024).

Algorithm:
  1. Sample N generations at T>0 (handled by the vLLM client; this module receives them).
  2. Greedy-cluster the N samples by bidirectional NLI entailment against each cluster's
     representative (the first member). O(N * |clusters|) NLI calls instead of O(N^2).
  3. p(c) = |c| / N; SE = -Σ p(c) log p(c).

High SE = answers disagree in MEANING -> likely hallucination.
Low SE  = answers agree in meaning -> likely confident.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

from hallucinations_math_lab.constants import LOG_EXTRA_KEY, DetectorCategory, DetectorName
from hallucinations_math_lab.custom_types.domain_types import Cluster, SampleSet
from hallucinations_math_lab.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorMetadata,
    DetectorPrerequisiteError,
    DetectorScore,
)
from hallucinations_math_lab.services.nli import NLIScorer, get_nli_scorer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticEntropyResult:
    semantic_entropy: float
    clusters: list[Cluster]


def cluster_by_bidirectional_entailment(
    answers: list[str],
    scorer: NLIScorer | None = None,
) -> list[Cluster]:
    """Greedy single-pass clustering by bidirectional NLI entailment.

    Each new answer is compared to the representative of each existing cluster. If it
    bidirectionally entails any representative, it joins that cluster; otherwise it
    starts a new one.
    """
    if not answers:
        raise ValueError("Cannot cluster empty answer list.")

    nli = scorer or get_nli_scorer()
    clusters: list[Cluster] = []

    for idx, ans in enumerate(answers):
        placed = False
        for cl in clusters:
            try:
                if nli.bidirectional_entails(cl.representative_text, ans):
                    cl.member_indices.append(idx)
                    placed = True
                    break
            except RuntimeError as e:
                logger.error(
                    "NLI entailment check failed; starting a new cluster as fallback.",
                    extra={LOG_EXTRA_KEY: {"error": str(e), "answer_index": idx}},
                )
                # Fail-fast on systemic NLI failure: re-raise after logging.
                raise
        if not placed:
            clusters.append(Cluster(member_indices=[idx], representative_text=ans))

    return clusters


def compute_semantic_entropy(sample_set: SampleSet, scorer: NLIScorer | None = None) -> SemanticEntropyResult:
    """Compute SE for a sample set. Raises if sample set is empty."""
    if not sample_set.samples:
        raise ValueError(
            f"Cannot compute SE: empty sample set for prompt={sample_set.prompt[:80]!r}"
        )

    answers = [s.text.strip() for s in sample_set.samples]
    clusters = cluster_by_bidirectional_entailment(answers, scorer=scorer)
    n_total = len(answers)

    probs = np.array([cl.size / n_total for cl in clusters], dtype=np.float64)
    # natural-log entropy (nats), matching the Nature paper
    se = float(-np.sum(probs * np.log(probs + 1e-12)))

    logger.info(
        "Computed semantic entropy.",
        extra={
            LOG_EXTRA_KEY: {
                "se_nats": round(se, 4),
                "n_clusters": len(clusters),
                "n_samples": n_total,
                "prompt": sample_set.prompt[:80],
            }
        },
    )

    return SemanticEntropyResult(semantic_entropy=se, clusters=clusters)


class SemanticEntropyDetector(Detector):
    name = DetectorName.SEMANTIC_ENTROPY
    requires_samples = True
    requires_hidden_state = False
    metadata = DetectorMetadata(
        signal="Shannon entropy over the cluster-size distribution of N samples grouped by bidirectional NLI entailment",
        assumption="when the model is hallucinating, independent samples diverge in meaning even when each individually sounds fluent",
        failure_modes=(
            "NLI noise on long-form answers — clustering becomes unreliable past a few sentences",
            "consistent paraphrase variation inflates SE even when meaning is stable",
            "fixed N=10 caps the resolution; small N can miss low-probability divergent meanings",
        ),
        category=DetectorCategory.UNCERTAINTY,
    )

    def __init__(self, scorer: NLIScorer | None = None) -> None:
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
        result = compute_semantic_entropy(ctx.samples, scorer=self._scorer)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DetectorScore(
            name=self.name,
            score=result.semantic_entropy,
            cost_usd=0.0,
            latency_ms=elapsed_ms,
            clusters=result.clusters,
        )
