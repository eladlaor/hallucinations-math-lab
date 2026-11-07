"""The default TriviaQA question sets: built on first use, reused afterwards.

No test here touches the network: the loader is injected.
"""

import json
from pathlib import Path

import pytest

from eval.datasets.default_question_sets import (
    BENCHMARK_SLICE,
    CALIBRATION_SLICE,
    QuestionSlice,
    ensure_question_set,
)
from eval.run_benchmark import build_parser as build_benchmark_parser
from eval.run_benchmark import resolve_questions_path


def _fake_loader(calls: list[tuple[int, int, Path]]):
    def load(n: int, offset: int, out: Path) -> list[dict]:
        calls.append((n, offset, out))
        rows = [{"id": f"q{offset + i}", "question": f"Q{offset + i}?"} for i in range(n)]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return rows

    return load


def test_missing_slice_is_built_once_then_reused(tmp_path: Path) -> None:
    calls: list[tuple[int, int, Path]] = []
    s = QuestionSlice(name="benchmark", offset=500, n=3, directory=tmp_path)

    first = ensure_question_set(s, loader=_fake_loader(calls))
    second = ensure_question_set(s, loader=_fake_loader(calls))

    assert first == second == s.path
    assert calls == [(3, 500, s.path)]
    assert len(s.path.read_text().splitlines()) == 3


def test_build_failure_names_the_slice(tmp_path: Path) -> None:
    def broken(n: int, offset: int, out: Path) -> list[dict]:
        raise OSError("network down")

    s = QuestionSlice(name="benchmark", offset=500, n=3, directory=tmp_path)
    with pytest.raises(RuntimeError, match=r"benchmark.*\[500, 503\)"):
        ensure_question_set(s, loader=broken)
    assert not s.path.exists()


def test_default_slices_never_share_questions() -> None:
    calibration = range(CALIBRATION_SLICE.offset, CALIBRATION_SLICE.offset + CALIBRATION_SLICE.n)
    benchmark = range(BENCHMARK_SLICE.offset, BENCHMARK_SLICE.offset + BENCHMARK_SLICE.n)
    assert not set(calibration) & set(benchmark)
    assert CALIBRATION_SLICE.path != BENCHMARK_SLICE.path


def test_default_slices_are_cached_under_gitignored_data_dir() -> None:
    for s in (CALIBRATION_SLICE, BENCHMARK_SLICE):
        assert s.path.parts[0] == "data"


def test_benchmark_without_in_uses_the_default_slice(tmp_path: Path) -> None:
    args = build_benchmark_parser().parse_args(["--out", "x.jsonl"])
    calls: list[tuple[int, int, Path]] = []
    s = QuestionSlice(name="benchmark", offset=500, n=2, directory=tmp_path)
    assert args.in_path is None
    assert resolve_questions_path(args.in_path, loader=_fake_loader(calls), slice_=s) == s.path
    assert calls == [(2, 500, s.path)]


def test_benchmark_with_in_uses_that_file_untouched(tmp_path: Path) -> None:
    own = tmp_path / "mine.jsonl"
    args = build_benchmark_parser().parse_args(["--in", str(own), "--out", "x.jsonl"])
    calls: list[tuple[int, int, Path]] = []
    assert resolve_questions_path(args.in_path, loader=_fake_loader(calls)) == own
    assert calls == []
