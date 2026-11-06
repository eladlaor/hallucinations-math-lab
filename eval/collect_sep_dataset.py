"""Collect the SEP training dataset: per-question SE labels and hidden states.

Each calibration question's N samples are clustered once by bidirectional NLI.
Both SE estimators are computed from that one clustering, so storing the
likelihood-weighted value next to the discrete one costs no extra NLI calls.
The probe trains on discrete SE (SEP_CLASSIFIER.md, "Which SE goes into the
label?"); the likelihood-weighted value is kept because it is the SE detector's
default score.
"""

from dataclasses import dataclass

from hallucinations_math_lab.constants import SemanticEntropyEstimator
from hallucinations_math_lab.custom_types.domain_types import SampleSet
from hallucinations_math_lab.services.nli import EntailmentScorer
from hallucinations_math_lab.services.semantic_entropy import compute_semantic_entropies

LABEL_ESTIMATORS = (
    SemanticEntropyEstimator.DISCRETE,
    SemanticEntropyEstimator.LIKELIHOOD_WEIGHTED,
)


@dataclass(frozen=True)
class SELabels:
    """One question's SE fields in the dataset row."""

    se_discrete: float
    se_likelihood_weighted: float
    cluster_ids: list[int]  # cluster index per sample, in sample order
    n_clusters: int


def label_sample_set(sample_set: SampleSet, scorer: EntailmentScorer | None = None) -> SELabels:
    """Discrete and likelihood-weighted SE, plus cluster ids, from one clustering.

    Raises before any NLI call if a sample lacks finite token logprobs.
    """
    result = compute_semantic_entropies(sample_set, LABEL_ESTIMATORS, scorer=scorer)
    cluster_ids = [0] * len(sample_set.samples)
    for cluster_id, cluster in enumerate(result.clusters):
        for member in cluster.member_indices:
            cluster_ids[member] = cluster_id
    return SELabels(
        se_discrete=result.entropies[SemanticEntropyEstimator.DISCRETE],
        se_likelihood_weighted=result.entropies[SemanticEntropyEstimator.LIKELIHOOD_WEIGHTED],
        cluster_ids=cluster_ids,
        n_clusters=len(result.clusters),
    )
