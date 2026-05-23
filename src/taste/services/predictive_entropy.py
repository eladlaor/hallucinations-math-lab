"""Token-level predictive entropy baseline.

This is the cheap signal: mean per-sample negative log-likelihood. It detects lexical
uncertainty but NOT semantic inconsistency. Implemented as a baseline against which
Semantic Entropy (semantic_entropy.py) is compared in the eval harness.
"""

import logging
import time

import numpy as np

from taste.constants import LOG_EXTRA_KEY, DetectorCategory, DetectorName
from taste.custom_types.domain_types import SampleSet
from taste.services.detector_protocol import (
    DetectionContext,
    Detector,
    DetectorMetadata,
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
    metadata = DetectorMetadata(
        signal="mean per-sample length-normalized negative log-likelihood",
        assumption="low token-level confidence correlates with incorrect answers",
        failure_modes=(
            "misses confident hallucinations where the model is wrong with high token probability",
            "noisy on very short answers where length normalization divides by a small T",
        ),
        category=DetectorCategory.UNCERTAINTY,
    )

    def score(self, ctx: DetectionContext) -> DetectorScore:
        if ctx.samples is None:
            raise DetectorPrerequisiteError(
                "PredictiveEntropyDetector requires samples; DetectionContext.samples is None."
            )
        # PE is a pure post-hoc computation over already-collected token
        # logprobs. The cost is amortized into whatever produced the samples;
        # the detector itself spends only python cycles, so cost_usd is 0.
        start = time.perf_counter()
        value = length_normalized_predictive_entropy(ctx.samples)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DetectorScore(
            name=self.name,
            score=value,
            cost_usd=0.0,
            latency_ms=elapsed_ms,
        )
