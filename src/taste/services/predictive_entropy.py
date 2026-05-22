"""Token-level predictive entropy baseline.

This is the cheap signal: mean per-sample negative log-likelihood. It detects lexical
uncertainty but NOT semantic inconsistency. Implemented as a baseline against which
Semantic Entropy (semantic_entropy.py) is compared in the eval harness.
"""

import logging

import numpy as np

from taste.constants import LOG_EXTRA_KEY, DetectorName
from taste.custom_types.domain_types import SampleSet
from taste.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorPrerequisiteError,
    DetectorScore,
)

logger = logging.getLogger(__name__)


def length_normalized_predictive_entropy(sample_set: SampleSet) -> float:
    """Mean across samples of the per-sample length-normalized negative log-likelihood.

    H_pred = (1/N) * Σ_i ( - (1/T_i) * Σ_t log p(token_{i,t}) )

    Higher = more uncertain at the token level.
    """
    if not sample_set.samples:
        raise ValueError("Cannot compute predictive entropy on empty sample set.")

    per_sample_nll: list[float] = []
    for i, s in enumerate(sample_set.samples):
        if not s.token_logprobs:
            logger.warning(
                "Sample missing token logprobs; skipping.",
                extra={LOG_EXTRA_KEY: {"sample_index": i, "prompt": sample_set.prompt[:80]}},
            )
            continue
        mean_lp = s.mean_token_logprob
        per_sample_nll.append(-mean_lp)

    if not per_sample_nll:
        raise ValueError("No samples had token logprobs; cannot compute predictive entropy.")

    return float(np.mean(per_sample_nll))


class PredictiveEntropyDetector(Detector):
    name = DetectorName.PREDICTIVE_ENTROPY
    requires_samples = True
    requires_hidden_state = False

    def score(self, ctx: DetectionContext) -> DetectorScore:
        if ctx.samples is None:
            raise DetectorPrerequisiteError(
                "PredictiveEntropyDetector requires samples; DetectionContext.samples is None."
            )
        return DetectorScore(
            name=self.name,
            score=length_normalized_predictive_entropy(ctx.samples),
        )
