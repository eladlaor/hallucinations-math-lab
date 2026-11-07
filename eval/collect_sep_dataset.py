"""SEP dataset collection: per-question SE labels from one NLI clustering.

Each calibration question's N samples are clustered once by bidirectional NLI.
Both SE estimators are computed from that one clustering, so storing the
likelihood-weighted value next to the discrete one costs no extra NLI calls.
The probe trains on discrete SE (SEP_CLASSIFIER.md, "Which SE goes into the
label?"); the likelihood-weighted value is kept because it is the SE detector's
default score.
"""

from dataclasses import dataclass

from hallucinations_math_lab.constants import SemanticEntropyEstimator
from hallucinations_math_lab.custom_types.domain_types import Cluster, SampleSet
from hallucinations_math_lab.services.nli import EntailmentScorer
from hallucinations_math_lab.services.semantic_entropy import compute_semantic_entropies

LABEL_ESTIMATORS = (
    SemanticEntropyEstimator.DISCRETE,
    SemanticEntropyEstimator.LIKELIHOOD_WEIGHTED,
)
_UNASSIGNED_CLUSTER_ID = -1


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
                    f"(prompt={sample_set.prompt[:80]!r})."
                )
            cluster_ids[member] = cluster_id
    missing = [i for i, cid in enumerate(cluster_ids) if cid == _UNASSIGNED_CLUSTER_ID]
    if missing:
        raise ValueError(
            f"Samples {missing} are in no cluster (prompt={sample_set.prompt[:80]!r})."
        )
    return cluster_ids
