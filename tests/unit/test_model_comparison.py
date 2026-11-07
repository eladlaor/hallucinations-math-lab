"""Multi-model evaluation: run_benchmark --sweep -> compare_models.

The sweep runs the same question set once per id in TARGET_MODEL_IDS, writing
one results file per model; compare_models puts those runs side by side and
refuses runs that are not comparable. Uses in-memory SLMs and string-equality
NLI, so it runs on CPU in milliseconds.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from demo.model_registry import get_model_entry
from eval.compare_models import comparison_report
from eval.compare_models import main as compare_models
from eval.run_benchmark import run as run_benchmark
from eval.run_benchmark import run_sweep
from hallucinations_math_lab import (
    DetectorName,
    PredictiveEntropyDetector,
    ResponseEvaluator,
    Sample,
    SampleSet,
    SemanticEntropyDetector,
    TokenLogprob,
)
from hallucinations_math_lab.config import get_settings
from hallucinations_math_lab.constants import LEGACY_SEP_PROBE_SUFFIX
from hallucinations_math_lab.custom_types.sep_probe_types import SEPProbeConfig
from hallucinations_math_lab.services.semantic_entropy_probe import (
    ObsoleteSEPProbeError,
    SEPProbeMismatchError,
    default_probe_dir,
    model_slug,
)
from tests.unit.sep_probe_fixtures import (
    PROBE_HIDDEN_SIZE,
    ProbeBackboneProvider,
    probe_config,
    write_probe_dir,
    write_train_ids,
)

SMALL_MODEL = "google/gemma-4-E2B-it"
MID_MODEL = "google/gemma-4-E4B-it"
QUESTIONS = [
    {"id": "q1", "question": "Capital of France?", "ground_truth_aliases": ["Paris"]},
    {"id": "q2", "question": "Author of 1984?", "ground_truth_aliases": ["Orwell"]},
    {"id": "q3", "question": "Largest moon of Neptune?", "ground_truth_aliases": ["Triton"]},
]
# What each fake model knows: the bigger one knows more.
KNOWLEDGE = {
    SMALL_MODEL: {"Capital of France?": "Paris"},
    MID_MODEL: {"Capital of France?": "Paris", "Author of 1984?": "Orwell"},
}


class FakeSLM(ProbeBackboneProvider):
    """Repeats the right answer for known questions, scatters otherwise; its
    backbone is the fixture probes' apart from the model id."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._known = KNOWLEDGE.get(model_id, {})

    def _answers(self, prompt: str) -> list[str]:
        if prompt in self._known:
            return [self._known[prompt]] * 10
        return [f"guess-{i}" for i in range(10)]

    def sample(self, prompt: str) -> SampleSet:
        return SampleSet(
            prompt=prompt,
            samples=[
                Sample(text=a, token_logprobs=[TokenLogprob(token=a, logprob=-0.5)])
                for a in self._answers(prompt)
            ],
        )

    def greedy(self, prompt: str) -> str:
        return self._answers(prompt)[0]

    def pooled_hidden_state(self, prompt: str, layers: tuple[int, ...]):
        return torch.zeros(len(layers), PROBE_HIDDEN_SIZE)


class EqualityNLI:
    def bidirectional_entails(self, a: str, b: str) -> bool:
        return a == b


def _evaluator(model_id: str) -> ResponseEvaluator:
    return ResponseEvaluator(
        slm=FakeSLM(model_id),
        detectors=[PredictiveEntropyDetector(), SemanticEntropyDetector(scorer=EqualityNLI())],
        escalation_detector=DetectorName.SEMANTIC_ENTROPY,
        escalation_threshold=1.0,
    )


@pytest.fixture
def settings_env(monkeypatch) -> Iterator[pytest.MonkeyPatch]:
    """Env-driven settings with a clean cache before and after."""
    get_settings.cache_clear()
    monkeypatch.setenv("TARGET_MODEL_IDS", json.dumps([SMALL_MODEL, MID_MODEL]))
    monkeypatch.delenv("SEP_PROBE_PATH", raising=False)
    yield monkeypatch
    get_settings.cache_clear()


def _questions_file(tmp_path: Path, rows: list[dict] = QUESTIONS) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_sweep_writes_one_results_file_per_model_one_model_at_a_time(tmp_path, settings_env):
    built: list[str] = []

    def factory(model_id: str) -> ResponseEvaluator:
        built.append(model_id)
        return _evaluator(model_id)

    paths = run_sweep(
        _questions_file(tmp_path), tmp_path / "out", judge=False, evaluator_factory=factory
    )

    assert built == [SMALL_MODEL, MID_MODEL]
    assert [p.name for p in paths] == ["google-gemma-4-e2b-it.jsonl", "google-gemma-4-e4b-it.jsonl"]
    for path, model_id in zip(paths, built, strict=True):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == len(QUESTIONS)
        assert {r["model_id"] for r in rows} == {model_id}


def test_sweep_rejects_an_unregistered_model_before_running_anything(tmp_path, settings_env):
    settings_env.setenv("TARGET_MODEL_IDS", json.dumps([SMALL_MODEL, "acme/not-a-real-model"]))

    with pytest.raises(ValueError, match="not in the model registry"):
        run_sweep(
            _questions_file(tmp_path), tmp_path / "out", judge=False, evaluator_factory=_evaluator
        )
    assert not (tmp_path / "out").exists()


def test_sweep_refuses_one_explicit_probe_path_for_several_models(tmp_path, settings_env):
    settings_env.setenv("ENABLED_DETECTORS", '["semantic_entropy_probe","semantic_entropy"]')
    settings_env.setenv("SEP_PROBE_PATH", str(tmp_path / "probe.pt"))

    with pytest.raises(ValueError, match="SEP_PROBE_PATH"):
        run_sweep(
            _questions_file(tmp_path), tmp_path / "out", judge=False, evaluator_factory=_evaluator
        )


def _registry_probe_config(model_id: str) -> SEPProbeConfig:
    """A probe config matching `model_id`'s registry entry, as a real probe would."""
    entry = get_model_entry(model_id)
    return probe_config(
        model_id=model_id,
        revision=entry.revision,
        num_hidden_layers=entry.num_hidden_layers,
        hidden_size=entry.hidden_size,
        layers=(entry.num_hidden_layers,),
    )


def _write_sweep_probe(model_id: str, config: SEPProbeConfig | None = None) -> Path:
    probe = write_probe_dir(default_probe_dir(model_id), config or _registry_probe_config(model_id))
    write_train_ids(probe, ["other"], model_id)
    return probe


def _refusing_factory(model_id: str) -> ResponseEvaluator:
    raise AssertionError(f"the sweep built an evaluator for {model_id} despite a bad probe")


def test_sweep_checks_every_models_probe_before_running_anything(tmp_path, settings_env):
    settings_env.setenv("ENABLED_DETECTORS", '["semantic_entropy_probe","semantic_entropy"]')
    settings_env.chdir(tmp_path)
    _write_sweep_probe(SMALL_MODEL)
    # No probe for MID_MODEL: the sweep must fail before scoring SMALL_MODEL.

    with pytest.raises(FileNotFoundError, match="gemma-4-e4b-it") as excinfo:
        run_sweep(
            _questions_file(tmp_path), tmp_path / "out", judge=False, evaluator_factory=_evaluator
        )
    assert not (tmp_path / "out").exists()
    message = str(excinfo.value)
    assert f"VLLM_MODEL_ID={MID_MODEL}" in message
    assert "ENABLED_DETECTORS" in message
    # A sweep refuses SEP_PROBE_PATH, so the hint must not suggest it.
    assert "SEP_PROBE_PATH" not in message


def test_preflight_expects_probe_dir_and_gives_pt_message(tmp_path, settings_env):
    """A legacy `.pt` probe where the default directory belongs is refused with the
    Ridge explanation, before any model is built."""
    settings_env.setenv("ENABLED_DETECTORS", '["semantic_entropy_probe","semantic_entropy"]')
    settings_env.chdir(tmp_path)
    _write_sweep_probe(SMALL_MODEL)
    legacy = (
        default_probe_dir(MID_MODEL).parent / f"{model_slug(MID_MODEL)}{LEGACY_SEP_PROBE_SUFFIX}"
    )
    legacy.write_bytes(b"pickled ridge")

    with pytest.raises(ObsoleteSEPProbeError, match=r"Ridge.*p\(high SE\)") as excinfo:
        run_sweep(
            _questions_file(tmp_path),
            tmp_path / "out",
            judge=False,
            evaluator_factory=_refusing_factory,
        )
    assert str(legacy) in str(excinfo.value)
    assert not (tmp_path / "out").exists()


def test_run_benchmark_preflight_fails_before_scoring_on_mismatch(tmp_path, settings_env):
    """A probe trained on another revision of the second model fails the sweep
    before the first model is scored, naming both values."""
    settings_env.setenv("ENABLED_DETECTORS", '["semantic_entropy_probe","semantic_entropy"]')
    settings_env.chdir(tmp_path)
    _write_sweep_probe(SMALL_MODEL)
    stale_revision = "f" * 40
    config = _registry_probe_config(MID_MODEL)
    stale_backbone = config.backbone.model_copy(update={"revision": stale_revision})
    _write_sweep_probe(MID_MODEL, config.model_copy(update={"backbone": stale_backbone}))

    with pytest.raises(SEPProbeMismatchError, match="revision") as excinfo:
        run_sweep(
            _questions_file(tmp_path),
            tmp_path / "out",
            judge=False,
            evaluator_factory=_refusing_factory,
        )
    message = str(excinfo.value)
    assert stale_revision in message
    assert get_model_entry(MID_MODEL).revision in message
    # A sweep refuses SEP_PROBE_PATH, so the hint must not suggest it.
    assert "SEP_PROBE_PATH" not in message
    assert f"VLLM_MODEL_ID={MID_MODEL}" in message
    assert not (tmp_path / "out").exists()


def test_sweep_preflight_accepts_registry_matching_probes(tmp_path, settings_env):
    settings_env.setenv("ENABLED_DETECTORS", '["semantic_entropy_probe","semantic_entropy"]')
    settings_env.chdir(tmp_path)
    _write_sweep_probe(SMALL_MODEL)
    _write_sweep_probe(MID_MODEL)

    paths = run_sweep(
        _questions_file(tmp_path), tmp_path / "out", judge=False, evaluator_factory=_evaluator
    )
    assert len(paths) == 2


def test_default_sep_probe_follows_the_providers_model_not_vllm_model_id(tmp_path, settings_env):
    settings_env.chdir(tmp_path)
    settings_env.setenv("VLLM_MODEL_ID", SMALL_MODEL)
    settings_env.setenv("ENABLED_DETECTORS", '["semantic_entropy_probe"]')
    settings_env.setenv("ESCALATION_DETECTOR", "semantic_entropy_probe")
    probe = write_probe_dir(default_probe_dir(MID_MODEL), probe_config(model_id=MID_MODEL))

    evaluator = ResponseEvaluator(slm=FakeSLM(MID_MODEL))

    (sep,) = evaluator.detectors
    assert sep.probe_dir == probe  # type: ignore[attr-defined]


def _results(tmp_path: Path, model_id: str, questions: list[dict] = QUESTIONS) -> Path:
    out = tmp_path / f"{model_id.replace('/', '_')}.jsonl"
    run_benchmark(
        _questions_file(tmp_path, questions), out, judge=False, evaluator=_evaluator(model_id)
    )
    return out


def test_comparison_reports_accuracy_and_auroc_per_model(tmp_path, settings_env, capsys):
    small, mid = _results(tmp_path, SMALL_MODEL), _results(tmp_path, MID_MODEL)

    report = comparison_report([small, mid])

    by_model = {m["model_id"]: m for m in report["models"]}
    assert by_model[SMALL_MODEL]["accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert by_model[MID_MODEL]["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert by_model[MID_MODEL]["detectors"]["semantic_entropy"]["auroc"] == 1.0
    assert report["n_questions"] == len(QUESTIONS)

    compare_models([small, mid], as_json=False)
    table = capsys.readouterr().out
    assert SMALL_MODEL in table and MID_MODEL in table and "semantic_entropy" in table


def test_comparison_refuses_runs_over_different_questions(tmp_path, settings_env):
    small = _results(tmp_path, SMALL_MODEL)
    mid = _results(tmp_path, MID_MODEL, questions=QUESTIONS[:2])

    with pytest.raises(RuntimeError, match="different questions"):
        comparison_report([small, mid])


def test_comparison_refuses_runs_with_different_sampling(tmp_path, settings_env):
    small = _results(tmp_path, SMALL_MODEL)
    settings_env.setenv("SAMPLE_N", "5")
    get_settings.cache_clear()
    mid = _results(tmp_path, MID_MODEL)

    with pytest.raises(RuntimeError, match="sampling"):
        comparison_report([small, mid])


def test_comparison_refuses_the_same_model_twice(tmp_path, settings_env):
    small = _results(tmp_path, SMALL_MODEL)
    copy = tmp_path / "copy.jsonl"
    copy.write_text(small.read_text())

    with pytest.raises(RuntimeError, match="more than once"):
        comparison_report([small, copy])
