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
from dataclasses import dataclass

import numpy as np

from taste.constants import LOG_EXTRA_KEY, DetectorName
from taste.custom_types.domain_types import Cluster, SampleSet
from taste.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorPrerequisiteError,
    DetectorScore,
)
from taste.services.nli import NLIScorer, get_nli_scorer

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

    def __init__(self, scorer: NLIScorer | None = None) -> None:
        self._scorer = scorer

    def score(self, ctx: DetectionContext) -> DetectorScore:
        if ctx.samples is None:
            raise DetectorPrerequisiteError(
                "SemanticEntropyDetector requires samples; DetectionContext.samples is None."
            )
        result = compute_semantic_entropy(ctx.samples, scorer=self._scorer)
        return DetectorScore(
            name=self.name,
            score=result.semantic_entropy,
            clusters=result.clusters,
        )
