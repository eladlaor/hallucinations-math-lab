"""GET /api/results and /api/results/report — the Measure the detectors tab (way in 3, read-only)."""

from pathlib import Path

import pytest

from eval.benchmark_results import load_rows
from eval.compare_models import comparison_report
from eval.compute_auroc import auroc_report
from tests.unit.web.conftest import ClientFactory, write_run

RUN_FILE = "triviaqa_200.jsonl"
SWEEP_DIR = "sweep"
SWEEP_MODELS = ("google/gemma-4-E2B-it", "google/gemma-4-E4B-it")


def _write_sweep(results_dir: Path) -> list[Path]:
    paths = []
    for i, model_id in enumerate(SWEEP_MODELS):
        path = results_dir / SWEEP_DIR / f"model_{i}.jsonl"
        write_run(path, model_id=model_id)
        paths.append(path)
    return paths


def test_listing_is_empty_when_there_are_no_results(
    make_client: ClientFactory, results_dir: Path
) -> None:
    body = make_client().get("/api/results").json()

    assert body == {"results_dir": str(results_dir.resolve()), "entries": []}


def test_listing_shows_single_runs_and_sweep_directories(
    make_client: ClientFactory, results_dir: Path
) -> None:
    write_run(results_dir / RUN_FILE)
    _write_sweep(results_dir)
    (results_dir / "notes.txt").write_text("not a results file")
    (results_dir / "empty_dir").mkdir()

    entries = make_client().get("/api/results").json()["entries"]

    assert entries == [
        {"path": SWEEP_DIR, "kind": "sweep", "n_files": 2},
        {"path": RUN_FILE, "kind": "run", "n_files": 1},
    ]


def test_run_report_equals_compute_auroc(make_client: ClientFactory, results_dir: Path) -> None:
    write_run(results_dir / RUN_FILE)

    body = make_client().get("/api/results/report", params={"path": RUN_FILE}).json()

    assert body["kind"] == "run"
    assert body["path"] == RUN_FILE
    assert body["report"] == auroc_report(load_rows(results_dir / RUN_FILE), source=RUN_FILE)


def test_sweep_report_equals_compare_models(make_client: ClientFactory, results_dir: Path) -> None:
    paths = _write_sweep(results_dir)

    body = make_client().get("/api/results/report", params={"path": SWEEP_DIR}).json()

    assert body["kind"] == "sweep"
    assert body["report"] == comparison_report(sorted(p.resolve() for p in paths))


@pytest.mark.parametrize("path", ["../../etc/passwd", "/etc/passwd", "sweep/../../outside.jsonl"])
def test_paths_outside_the_results_directory_are_400(
    make_client: ClientFactory, results_dir: Path, path: str
) -> None:
    _write_sweep(results_dir)
    (results_dir.parent / "outside.jsonl").write_text("{}\n")

    response = make_client().get("/api/results/report", params={"path": path})

    assert response.status_code == 400


def test_missing_results_path_is_404(make_client: ClientFactory, results_dir: Path) -> None:
    results_dir.mkdir()

    response = make_client().get("/api/results/report", params={"path": "nope.jsonl"})

    assert response.status_code == 404


def test_an_unscorable_results_file_is_422_with_the_reason(
    make_client: ClientFactory, results_dir: Path
) -> None:
    results_dir.mkdir()
    (results_dir / "broken.jsonl").write_text("{not json\n")

    response = make_client().get("/api/results/report", params={"path": "broken.jsonl"})

    assert response.status_code == 422
    assert "not valid JSON" in response.json()["detail"]
