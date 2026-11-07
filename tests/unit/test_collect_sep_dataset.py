"""SEP dataset collection: per-question SE labels and the dataset directory.

Contract pinned here (SEP_CLASSIFIER.md, "Which SE goes into the label?" and
"Artifact format"):
  - The samples are clustered once; discrete and likelihood-weighted SE are both
    computed from that one clustering, so no extra NLI calls are made.
  - The row carries both values, the per-sample cluster ids and the cluster count.
  - Collection always writes rows.parquet, hidden_states.safetensors (every
    layer at TBG and SLT) and collection_config.json, plus one training-log
    line per question. The config names the calibration set and its SHA-256.
  - Only a provider with every-layer extraction (HF transformers) collects, and
    the backbone revision and depth must be the registry's; all are checked
    before any sampling.
  - A failed question stops the run, and the questions finished before it are
    still written as a loadable dataset.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

import eval.collect_sep_dataset as collect_module
from demo.model_registry import get_model_entry
from demo.slm_provider import SamplingTruncation
from eval.collect_sep_dataset import (
    CalibrationQuestion,
    CalibrationSet,
    cluster_ids_per_sample,
    collect_sep_dataset,
    label_sample_set,
    read_calibration_set,
)
from eval.sep_dataset_io import SEPDataset, TrainingLogEntry, load_sep_dataset
from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import (
    SEP_DATASET_ROWS_FILE,
    SEP_DATASET_TENSORS_FILE,
    SEP_TRAINING_LOG_FILE,
    TokenPosition,
)
from hallucinations_math_lab.custom_types.domain_types import (
    Cluster,
    Sample,
    SampleSet,
    TokenLogprob,
)
from hallucinations_math_lab.services.semantic_entropy import (
    cluster_by_bidirectional_entailment,
    compute_semantic_entropy,
)


class CountingNLI:
    """Bidirectional entailment iff the two strings are equal; counts every call."""

    def __init__(self) -> None:
        self.calls = 0

    def bidirectional_entails(self, a: str, b: str) -> bool:
        self.calls += 1
        return a == b


def _sample(text: str, logprob: float) -> Sample:
    return Sample(text=text, token_logprobs=[TokenLogprob(token="t", logprob=logprob)])


# 6 "Paris", 3 "London", 1 "Berlin", with unequal likelihoods so the two estimators differ.
_SAMPLE_SET = SampleSet(
    prompt="What is the capital of France?",
    samples=[
        _sample("Paris", -0.1),
        _sample("London", -2.0),
        _sample("Paris", -0.2),
        _sample("Berlin", -3.0),
        _sample("Paris", -0.1),
        _sample("London", -1.5),
        _sample("Paris", -0.3),
        _sample("Paris", -0.1),
        _sample("London", -2.5),
        _sample("Paris", -0.2),
    ],
)


def test_collector_row_has_both_estimators_from_one_clustering() -> None:
    one_clustering = CountingNLI()
    cluster_by_bidirectional_entailment(
        [s.text for s in _SAMPLE_SET.samples], scorer=one_clustering, question=_SAMPLE_SET.prompt
    )

    nli = CountingNLI()
    labels = label_sample_set(_SAMPLE_SET, scorer=nli)

    assert nli.calls == one_clustering.calls
    assert labels.se_discrete == pytest.approx(
        -(0.6 * math.log(0.6) + 0.3 * math.log(0.3) + 0.1 * math.log(0.1))
    )
    weighted = compute_semantic_entropy(_SAMPLE_SET, scorer=CountingNLI()).semantic_entropy
    assert labels.se_likelihood_weighted == pytest.approx(weighted)
    assert labels.se_likelihood_weighted != pytest.approx(labels.se_discrete)
    assert labels.cluster_ids == [0, 1, 0, 2, 0, 1, 0, 0, 1, 0]
    assert labels.n_clusters == 3


def test_collector_validates_logprobs_before_any_nli_call() -> None:
    sample_set = SampleSet(prompt="q", samples=[Sample(text="Paris"), Sample(text="Lyon")])
    nli = CountingNLI()
    with pytest.raises(ValueError, match="token logprobs"):
        label_sample_set(sample_set, scorer=nli)
    assert nli.calls == 0


def test_cluster_ids_reject_a_sample_in_no_cluster() -> None:
    sample_set = SampleSet(prompt="q", samples=[Sample(text="a"), Sample(text="b")])
    clusters = [Cluster(member_indices=[0], representative_text="a")]
    with pytest.raises(ValueError, match=r"Samples \[1\] are in no cluster"):
        cluster_ids_per_sample(clusters, sample_set)


def test_cluster_ids_reject_a_sample_in_two_clusters() -> None:
    sample_set = SampleSet(prompt="q", samples=[Sample(text="a"), Sample(text="b")])
    clusters = [
        Cluster(member_indices=[0, 1], representative_text="a"),
        Cluster(member_indices=[1], representative_text="b"),
    ]
    with pytest.raises(ValueError, match="Sample 1 is in clusters 0 and 1"):
        cluster_ids_per_sample(clusters, sample_set)


# --- Dataset collection (SEP_CLASSIFIER.md item 9) ---------------------------

E2B_ID = "google/gemma-4-E2B-it"
_E2B = get_model_entry(E2B_ID)
_N_STATES = _E2B.num_hidden_layers + 1
_FAKE_TEMPLATE_SHA = "ab" * 32
# Gemma 4's generation_config values, which `generate` applies under the temperature.
_FAKE_TRUNCATION = SamplingTruncation(top_p=0.95, top_k=64)
# The fake provider writes `question index + offset` into every hidden value,
# so a read-back shows which question and which position a row came from.
_POSITION_OFFSET = {TokenPosition.TBG: 0.0, TokenPosition.SLT: 0.5}

_QUESTIONS = [
    CalibrationQuestion(id="q0", question="Capital of France?", ground_truth_aliases=["Paris"]),
    CalibrationQuestion(id="q1", question="Capital of Peru?", ground_truth_aliases=["Lima"]),
    CalibrationQuestion(id="q2", question="Capital of Chad?", ground_truth_aliases=["N'Djamena"]),
]
_CALIBRATION_SET = CalibrationSet(name="questions.jsonl", sha256="cd" * 32, questions=_QUESTIONS)
_GREEDY = {"Capital of France?": "Paris", "Capital of Peru?": "Cusco", "Capital of Chad?": "Paris"}


class _FakeCollectionProvider:
    """Stands in for HFTransformersProvider: samples, a greedy answer and every
    layer's hidden state at TBG and SLT, in registry-sized tensors."""

    def __init__(
        self,
        revision: str = _E2B.revision,
        hidden_states: bool = True,
        num_hidden_layers: int = _E2B.num_hidden_layers,
        n_states: int = _N_STATES,
        greedy_failure_prompt: str | None = None,
        greedy_failure: BaseException | None = None,
    ) -> None:
        self.model_id = E2B_ID
        self._revision = revision
        self._hidden_states = hidden_states
        self._num_hidden_layers = num_hidden_layers
        self._n_states = n_states
        self._greedy_failure_prompt = greedy_failure_prompt
        self._greedy_failure = greedy_failure or ValueError("Answer '' encodes to no tokens.")
        self.sample_calls = 0
        self.hidden_state_calls: list[tuple[str, TokenPosition, str | None]] = []

    @property
    def model_revision(self) -> str:
        return self._revision

    @property
    def num_hidden_layers(self) -> int:
        return self._num_hidden_layers

    @property
    def chat_template_sha256(self) -> str:
        return _FAKE_TEMPLATE_SHA

    @property
    def sampling_truncation(self) -> SamplingTruncation:
        return _FAKE_TRUNCATION

    def sample(self, prompt: str) -> SampleSet:
        self.sample_calls += 1
        return SampleSet(prompt=prompt, samples=_SAMPLE_SET.samples)

    def greedy(self, prompt: str) -> str:
        if prompt == self._greedy_failure_prompt:
            raise self._greedy_failure
        return _GREEDY[prompt]

    def pooled_hidden_state(self, prompt: str, layers: tuple[int, ...]) -> torch.Tensor:
        raise AssertionError("collection reads every layer, not pooled rows")

    def hidden_states_all_layers(
        self, prompt: str, position: TokenPosition, answer: str | None = None
    ) -> torch.Tensor | None:
        self.hidden_state_calls.append((prompt, position, answer))
        if not self._hidden_states:
            return None
        index = [q.question for q in _QUESTIONS].index(prompt)
        return torch.full((self._n_states, _E2B.hidden_size), index + _POSITION_OFFSET[position])


def _collect(tmp_path: Path, provider: object) -> SEPDataset:
    return collect_sep_dataset(
        provider, _CALIBRATION_SET, tmp_path / "dataset", scorer=CountingNLI()
    )


def test_dataset_rows_and_columns(tmp_path: Path) -> None:
    provider = _FakeCollectionProvider()
    collected = _collect(tmp_path, provider)
    dataset = load_sep_dataset(tmp_path / "dataset")

    assert dataset.rows == collected.rows
    assert [r.id for r in dataset.rows] == ["q0", "q1", "q2"]
    first = dataset.rows[0]
    assert first.question == "Capital of France?"
    assert first.ground_truth_aliases == ["Paris"]
    assert first.sampled_answers == [s.text for s in _SAMPLE_SET.samples]
    assert first.cluster_ids == [0, 1, 0, 2, 0, 1, 0, 0, 1, 0]
    assert first.n_clusters == 3
    assert first.se_discrete == pytest.approx(
        -(0.6 * math.log(0.6) + 0.3 * math.log(0.3) + 0.1 * math.log(0.1))
    )
    assert [r.greedy_answer for r in dataset.rows] == ["Paris", "Cusco", "Paris"]
    assert [r.greedy_correct for r in dataset.rows] == [True, False, False]
    # SLT teacher-forces the greedy answer; TBG takes none.
    assert ("Capital of Peru?", TokenPosition.SLT, "Cusco") in provider.hidden_state_calls
    assert ("Capital of Peru?", TokenPosition.TBG, None) in provider.hidden_state_calls

    for position, offset in _POSITION_OFFSET.items():
        states = dataset.hidden_states[position]
        assert states.shape == (len(_QUESTIONS), _N_STATES, _E2B.hidden_size)
        assert states.dtype == np.float32
        assert states[:, 0, 0].tolist() == [0 + offset, 1 + offset, 2 + offset]

    columns = pq.read_schema(tmp_path / "dataset" / SEP_DATASET_ROWS_FILE).names
    assert columns == [
        "id",
        "question",
        "ground_truth_aliases",
        "sampled_answers",
        "cluster_ids",
        "se_discrete",
        "se_likelihood_weighted",
        "n_clusters",
        "greedy_answer",
        "greedy_correct",
    ]
    settings = get_settings()
    config = dataset.config
    assert config.calibration_set == _CALIBRATION_SET.name
    assert config.calibration_set_sha256 == _CALIBRATION_SET.sha256
    assert config.n_questions_requested == len(_QUESTIONS)
    assert config.backbone.model_id == E2B_ID
    assert config.backbone.revision == _E2B.revision
    assert config.backbone.num_hidden_layers == _E2B.num_hidden_layers
    assert config.backbone.hidden_size == _E2B.hidden_size
    assert config.backbone.dtype == _E2B.dtype
    assert config.prompt.chat_template_sha256 == _FAKE_TEMPLATE_SHA
    assert config.prompt.add_generation_prompt is True
    assert config.prompt.answer_instruction == settings.answer_instruction
    assert config.labels.sample_n == settings.sample_n
    assert config.labels.temperature == settings.sample_temperature
    # The effective truncation, not "none sent": generate applies the checkpoint's values.
    assert config.labels.top_p == _FAKE_TRUNCATION.top_p
    assert config.labels.top_k == _FAKE_TRUNCATION.top_k
    assert config.labels.max_tokens == settings.max_tokens
    assert config.labels.nli_model_id == settings.nli_model_id
    assert config.labels.nli_revision == settings.nli_revision


class _PooledOnlyProvider(_FakeCollectionProvider):
    """Like the vLLM provider: no every-layer extraction."""

    hidden_states_all_layers = None  # type: ignore[assignment]


def test_collect_requires_hf_provider(tmp_path: Path) -> None:
    """A provider without every-layer extraction (vLLM) is refused before any sampling."""
    provider = _PooledOnlyProvider()
    with pytest.raises(TypeError, match="SLM_PROVIDER=hf_transformers"):
        _collect(tmp_path, provider)
    assert provider.sample_calls == 0
    assert not (tmp_path / "dataset").exists()


@pytest.mark.parametrize("revision", ["", "main", "0" * 40])
def test_collect_fails_fast_on_missing_or_wrong_revision(tmp_path: Path, revision: str) -> None:
    provider = _FakeCollectionProvider(revision=revision)
    with pytest.raises(ValueError, match=_E2B.revision):
        _collect(tmp_path, provider)
    assert provider.sample_calls == 0


def test_collect_fails_fast_on_a_depth_that_disagrees_with_the_registry(tmp_path: Path) -> None:
    provider = _FakeCollectionProvider(num_hidden_layers=_E2B.num_hidden_layers + 1)
    with pytest.raises(
        ValueError,
        match=rf"{_E2B.num_hidden_layers + 1} decoder layers.*has {_E2B.num_hidden_layers}",
    ):
        _collect(tmp_path, provider)
    assert provider.sample_calls == 0


def test_collect_fails_fast_on_hidden_states_without_the_embedding_row(tmp_path: Path) -> None:
    """[L, d] instead of [L+1, d]: a dropped embedding row would shift every layer index."""
    with pytest.raises(RuntimeError, match=r"q0.*\(L\+1, hidden_size\)"):
        _collect(tmp_path, _FakeCollectionProvider(n_states=_E2B.num_hidden_layers))
    assert not (tmp_path / "dataset" / SEP_DATASET_TENSORS_FILE).exists()


def test_collect_writes_the_finished_questions_before_failing(tmp_path: Path) -> None:
    """A late failure stops the run but keeps the hours of collection before it."""
    provider = _FakeCollectionProvider(greedy_failure_prompt="Capital of Chad?")
    with pytest.raises(RuntimeError, match=r"'q2' \(3/3.*the 2 questions before it.*no tokens"):
        _collect(tmp_path, provider)

    with pytest.raises(ValueError, match="incomplete: 2 of 3"):
        load_sep_dataset(tmp_path / "dataset")
    dataset = load_sep_dataset(tmp_path / "dataset", allow_incomplete=True)
    assert [r.id for r in dataset.rows] == ["q0", "q1"]
    for position, offset in _POSITION_OFFSET.items():
        assert dataset.hidden_states[position][:, 0, 0].tolist() == [0 + offset, 1 + offset]
    log = (tmp_path / "dataset" / SEP_TRAINING_LOG_FILE).read_text().splitlines()
    assert len(log) == 2


def test_collect_writes_the_finished_questions_when_interrupted(tmp_path: Path) -> None:
    """Ctrl-C is how a long GPU run is stopped: it keeps the rows and stays a KeyboardInterrupt."""
    provider = _FakeCollectionProvider(
        greedy_failure_prompt="Capital of Chad?", greedy_failure=KeyboardInterrupt()
    )
    with pytest.raises(KeyboardInterrupt):
        _collect(tmp_path, provider)
    dataset = load_sep_dataset(tmp_path / "dataset", allow_incomplete=True)
    assert [r.id for r in dataset.rows] == ["q0", "q1"]


def test_a_failed_salvage_write_still_names_the_failed_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_write(directory: Path, dataset: SEPDataset) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(collect_module, "write_sep_dataset", failing_write)
    provider = _FakeCollectionProvider(greedy_failure_prompt="Capital of Chad?")
    with pytest.raises(RuntimeError, match=r"'q2'.*no tokens.*2 questions before it.*disk full"):
        _collect(tmp_path, provider)


def test_collect_writes_the_finished_questions_when_the_log_append_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after a question's collection, here its log line, still keeps the earlier rows."""
    appended: list[str] = []

    def append_or_fail(directory: Path, entry: TrainingLogEntry) -> None:
        if entry.id == "q2":
            raise RuntimeError("disk full")
        appended.append(entry.id)

    monkeypatch.setattr(collect_module, "append_training_log", append_or_fail)
    with pytest.raises(RuntimeError, match=r"'q2'.*the 2 questions before it.*disk full"):
        _collect(tmp_path, _FakeCollectionProvider())
    dataset = load_sep_dataset(tmp_path / "dataset", allow_incomplete=True)
    assert [r.id for r in dataset.rows] == appended == ["q0", "q1"]
    for position, offset in _POSITION_OFFSET.items():
        assert dataset.hidden_states[position][:, 0, 0].tolist() == [0 + offset, 1 + offset]


def test_collect_fails_fast_on_missing_hidden_states(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"q0.*no hidden states"):
        _collect(tmp_path, _FakeCollectionProvider(hidden_states=False))
    assert not (tmp_path / "dataset" / SEP_DATASET_TENSORS_FILE).exists()


def test_collect_refuses_a_non_empty_output_dir(tmp_path: Path) -> None:
    """A finished collection costs hours of GPU time; it is never overwritten."""
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / SEP_DATASET_ROWS_FILE).write_text("old")
    provider = _FakeCollectionProvider(revision="")
    # Checked before the config: reading the template would load the model.
    with pytest.raises(FileExistsError, match="not empty"):
        _collect(tmp_path, provider)
    assert provider.sample_calls == 0


def test_training_log_one_line_per_question(tmp_path: Path) -> None:
    _collect(tmp_path, _FakeCollectionProvider())
    lines = (tmp_path / "dataset" / SEP_TRAINING_LOG_FILE).read_text().splitlines()

    assert len(lines) == len(_QUESTIONS)
    entries = [json.loads(line) for line in lines]
    assert [e["id"] for e in entries] == ["q0", "q1", "q2"]
    assert set(entries[0]) == {
        "id",
        "se_discrete",
        "se_likelihood_weighted",
        "n_clusters",
        "sampled_answer_word_counts",
        "greedy_answer_word_count",
        "greedy_correct",
        "elapsed_s",
    }
    assert entries[0]["sampled_answer_word_counts"] == [1] * len(_SAMPLE_SET.samples)
    assert entries[1]["greedy_correct"] is False
    assert all(e["elapsed_s"] >= 0 for e in entries)


def test_read_calibration_set_requires_aliases(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"id": "q0", "question": "Capital of France?"}) + "\n")
    with pytest.raises(ValueError, match=r"line 1.*ground_truth_aliases"):
        read_calibration_set(path)


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ('{"id": "q0", "question": null, "ground_truth_aliases": ["A"]}', "question must be"),
        ('{"id": "q0", "question": "  ", "ground_truth_aliases": ["A"]}', "question must be"),
        ('{"id": null, "question": "Q?", "ground_truth_aliases": ["A"]}', "id must be"),
        ('{"id": true, "question": "Q?", "ground_truth_aliases": ["A"]}', "id must be"),
        ('{"id": "q0", "question": "Q?", "ground_truth_aliases": [null]}', "list of strings"),
        ('["q0", "Q?"]', "JSON object, got list"),
    ],
)
def test_read_calibration_set_rejects_wrong_types(tmp_path: Path, line: str, match: str) -> None:
    """Nothing is coerced: a null question would otherwise be collected as the prompt "None"."""
    path = tmp_path / "questions.jsonl"
    path.write_text(line + "\n")
    with pytest.raises(ValueError, match=rf"questions.jsonl line 1.*{match}"):
        read_calibration_set(path)


def test_read_calibration_set_accepts_integer_ids(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"id": 7, "question": "Q?", "ground_truth_aliases": ["A"]}) + "\n")
    assert read_calibration_set(path).questions == [
        CalibrationQuestion(id="7", question="Q?", ground_truth_aliases=["A"])
    ]


def test_read_calibration_set_names_and_hashes_the_file(tmp_path: Path) -> None:
    """The fit reads only the dataset directory, so the config must identify the questions."""
    path = tmp_path / "calibration.jsonl"
    path.write_text(json.dumps({"id": "q0", "question": "Q?", "ground_truth_aliases": ["A"]}))
    calibration_set = read_calibration_set(path)
    assert calibration_set.name == "calibration.jsonl"
    assert calibration_set.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_calibration_set_rejects_duplicate_ids(tmp_path: Path) -> None:
    row = {"id": "q0", "question": "Q?", "ground_truth_aliases": ["A"]}
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate question ids"):
        read_calibration_set(path)
