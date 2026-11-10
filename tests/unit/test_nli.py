"""NLIScorer label handling, with a fake model so no checkpoint is downloaded.

Contract pinned here:
  - Label names come from the model's own `config.id2label`, whatever order
    the checkpoint uses; there is no hardcoded index order.
  - `label_probs` is a softmax over the three labels and sums to 1.
  - `classify` is the argmax of `label_probs`.
  - `bidirectional_entails` is `classify` entailment in both directions (what
    the SE trace relies on).
  - A label that is not entailment / neutral / contradiction fails fast and
    names the model id.
  - The tokenizer and model load at the pinned `nli_revision` commit.
  - An explicit model_id is accepted only with its revision, a 40-hex commit sha.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import NLILabel
from hallucinations_math_lab.services import nli
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
    scorer = NLIScorer(model_id="fake/nli", revision="f" * 40)
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


class _OneWayModel:
    """Entailment iff the premise is contained in the hypothesis, so the two
    directions can disagree ("Paris" entails "Paris, France", not the reverse)."""

    config = SimpleNamespace(id2label=_UNUSUAL_ORDER)

    def __call__(self, *, premise: str, hypothesis: str) -> SimpleNamespace:
        logits = [3.0, 1.0, 0.0] if premise in hypothesis else [0.0, 3.0, 1.0]
        return SimpleNamespace(logits=torch.tensor([logits]))


def test_bidirectional_entails_is_classify_both_ways():
    """The SE trace joins clusters on two `classify` calls; that must be exactly
    `bidirectional_entails` (the `DirectionalEntailmentScorer` contract)."""
    scorer = NLIScorer(model_id="fake/nli", revision="f" * 40)
    scorer._tokenizer = lambda premise, hypothesis, **_k: _FakeEncoding(  # type: ignore[assignment]
        premise=premise, hypothesis=hypothesis
    )
    scorer._model = _OneWayModel()  # type: ignore[assignment]
    texts = ["Paris", "Paris, France", "Rome"]

    for a in texts:
        for b in texts:
            both = (
                scorer.classify(a, b) is NLILabel.ENTAILMENT
                and scorer.classify(b, a) is NLILabel.ENTAILMENT
            )
            assert scorer.bidirectional_entails(a, b) is both
    assert scorer.classify("Paris", "Paris, France") is NLILabel.ENTAILMENT
    assert not scorer.bidirectional_entails("Paris", "Paris, France")


def test_unmapped_label_fails_fast_and_names_the_model():
    with pytest.raises(ValueError, match="fake/nli"):
        nli_label_by_index({0: "LABEL_0", 1: "neutral", 2: "contradiction"}, "fake/nli")


def test_missing_label_fails_fast():
    with pytest.raises(ValueError, match="entailment"):
        nli_label_by_index({0: "neutral", 1: "contradiction"}, "fake/nli")


def test_nli_loads_at_the_pinned_revision(monkeypatch):
    """SE labels depend on the NLI weights, so they load from one immutable commit."""
    tokenizer_spy = MagicMock()
    model_spy = MagicMock()
    monkeypatch.setattr(nli.AutoTokenizer, "from_pretrained", tokenizer_spy)
    monkeypatch.setattr(nli.AutoModelForSequenceClassification, "from_pretrained", model_spy)

    NLIScorer()._ensure_loaded()

    settings = get_settings()
    assert re.fullmatch(r"[0-9a-f]{40}", settings.nli_revision)
    for spy in (tokenizer_spy, model_spy):
        assert spy.call_args.args == (settings.nli_model_id,)
        assert spy.call_args.kwargs["revision"] == settings.nli_revision


@pytest.mark.parametrize(
    ("model_id", "revision"), [("fake/nli", None), (None, "f" * 40)], ids=["no-rev", "no-model"]
)
def test_nli_refuses_a_model_without_its_revision(model_id: str | None, revision: str | None):
    """An explicit model is pinned too; a lone model_id would load the Hub's default branch."""
    with pytest.raises(ValueError, match="model_id and revision together"):
        NLIScorer(model_id=model_id, revision=revision)


@pytest.mark.parametrize("revision", ["main", "v1.0", "F" * 40, "f" * 39])
def test_nli_refuses_an_explicit_revision_that_is_not_a_commit_sha(revision: str):
    """A branch or tag can move, so an explicit model is pinned to a commit like the default."""
    with pytest.raises(ValueError, match=r"fake/nli.*40-hex"):
        NLIScorer(model_id="fake/nli", revision=revision)
