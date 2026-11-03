"""NLIScorer label handling, with a fake model so no checkpoint is downloaded.

Contract pinned here:
  - Label names come from the model's own `config.id2label`, whatever order
    the checkpoint uses; there is no hardcoded index order.
  - `label_probs` is a softmax over the three labels and sums to 1.
  - `classify` is the argmax of `label_probs`.
  - A label that is not entailment / neutral / contradiction fails fast and
    names the model id.
"""

from types import SimpleNamespace

import pytest
import torch

from hallucinations_math_lab.constants import NLILabel
from hallucinations_math_lab.services.nli import NLIScorer, nli_label_by_index


class _FakeEncoding(dict):
    def to(self, _device: str) -> "_FakeEncoding":
        return self


class _FakeModel:
    """Returns fixed logits; `id2label` order is deliberately not MNLI's."""

    def __init__(self, logits: list[float], id2label: dict[int, str]) -> None:
        self.config = SimpleNamespace(id2label=id2label)
        self._logits = torch.tensor([logits])

    def __call__(self, **_inputs: object) -> SimpleNamespace:
        return SimpleNamespace(logits=self._logits)


def _scorer(logits: list[float], id2label: dict[int, str]) -> NLIScorer:
    scorer = NLIScorer(model_id="fake/nli")
    scorer._tokenizer = lambda *_a, **_k: _FakeEncoding()  # type: ignore[assignment]
    scorer._model = _FakeModel(logits, id2label)  # type: ignore[assignment]
    return scorer


_UNUSUAL_ORDER = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}


def test_label_by_index_reads_the_model_config_case_insensitively():
    assert nli_label_by_index(_UNUSUAL_ORDER, "fake/nli") == {
        0: NLILabel.ENTAILMENT,
        1: NLILabel.NEUTRAL,
        2: NLILabel.CONTRADICTION,
    }


def test_label_probs_follow_the_config_order_and_sum_to_one():
    probs = _scorer([3.0, 1.0, 0.0], _UNUSUAL_ORDER).label_probs("a", "b")
    assert set(probs) == set(NLILabel)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert max(probs, key=lambda label: probs[label]) is NLILabel.ENTAILMENT
    assert probs[NLILabel.ENTAILMENT] > probs[NLILabel.NEUTRAL] > probs[NLILabel.CONTRADICTION]


def test_classify_is_the_argmax_of_label_probs():
    assert _scorer([3.0, 1.0, 0.0], _UNUSUAL_ORDER).classify("a", "b") is NLILabel.ENTAILMENT
    assert _scorer([0.0, 1.0, 3.0], _UNUSUAL_ORDER).classify("a", "b") is NLILabel.CONTRADICTION


def test_unmapped_label_fails_fast_and_names_the_model():
    with pytest.raises(ValueError, match="fake/nli"):
        nli_label_by_index({0: "LABEL_0", 1: "neutral", 2: "contradiction"}, "fake/nli")


def test_missing_label_fails_fast():
    with pytest.raises(ValueError, match="entailment"):
        nli_label_by_index({0: "neutral", 1: "contradiction"}, "fake/nli")
