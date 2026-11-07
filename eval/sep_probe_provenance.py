"""Which SEP probe a benchmark ran, and a guard against evaluating it on its training set.

A probe scored on its own training prompts reports an optimistic AUROC: the
linear map has seen those exact hidden states. train_sep_probe.py writes the
training question ids into the probe directory (`<probe dir>/train_ids.json`,
an `SEPTrainIds`), and the detector loads them with the weights;
run_benchmark.py refuses to score any of those ids with that probe.

`probe_run_record` is what run_benchmark.py records on every row about the
probe in use: the reference it was resolved from, the Hub commit when it came
from the Hub, and the SHA-256 of the two files that decide its scores, as the
detector loaded them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

from hallucinations_math_lab.constants import SEP_TRAIN_IDS_FILE
from hallucinations_math_lab.services.semantic_entropy_probe import SemanticEntropyProbeDetector

# How many overlapping ids to name in the error message.
MAX_IDS_IN_ERROR: Final[int] = 5


@dataclass(frozen=True)
class SEPProbeRunRecord:
    """The probe behind a benchmark run. `ref` is the probe directory as given (never an
    absolute path the code made up) or the `hf://` URI; `revision` is the Hub commit downloaded, None for a
    local directory; the hashes pin the exact weights and config that were scored."""

    ref: str
    revision: str | None
    weights_sha256: str
    config_sha256: str


def probe_run_record(probe: SemanticEntropyProbeDetector) -> dict[str, str | None]:
    """The JSON-ready provenance of the probe `probe` scores with. The hashes are of the
    bytes it loaded, not of the directory now, which may have been retrained since."""
    return asdict(
        SEPProbeRunRecord(
            ref=probe.source.ref,
            revision=probe.source.revision,
            weights_sha256=probe.weights_sha256,
            config_sha256=probe.config_sha256,
        )
    )


def assert_disjoint(probe: SemanticEntropyProbeDetector, benchmark_ids: list[str]) -> None:
    """Fail fast if any benchmark question was used to train `probe`. Checks the training
    ids the detector loaded with its weights, never the directory as it is now, which may
    have been retrained since."""
    if probe.train_ids is None:
        raise RuntimeError(
            f"SEP probe {probe.probe_dir} had no training-id file ({SEP_TRAIN_IDS_FILE}) when "
            "it was loaded, so it cannot be shown not to overlap the benchmark. Retrain it "
            "with eval.train_sep_probe."
        )
    overlap = sorted(set(benchmark_ids) & set(probe.train_ids.question_ids))
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} benchmark questions were used to train the SEP probe "
            f"{probe.probe_dir} (e.g. {overlap[:MAX_IDS_IN_ERROR]}); its scores on them would "
            "be optimistic. Build the benchmark from a disjoint slice (load_triviaqa --offset)."
        )
