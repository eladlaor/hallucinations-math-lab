"""Compare models side by side from the results files of `run_benchmark --sweep`.

Each input file is one model's run over the same question set. For each model
the report gives accuracy (greedy answers that contain an accepted answer) and
one AUROC per detector with its bootstrap 95% CI, exactly as compute_auroc.py
computes them for a single run.

The runs must be comparable, so the script fails fast unless every file:
  - comes from a different model,
  - used the same sampling settings (sample_n, sample_temperature),
  - covers the same questions.

Output: a table by default, the full JSON with `--json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

from eval.benchmark_results import (
    ID_FIELD,
    MODEL_ID_FIELD,
    SAMPLE_N_FIELD,
    SAMPLE_TEMPERATURE_FIELD,
    BenchmarkRow,
    load_rows,
)
from eval.compute_auroc import REPORT_DECIMALS, auroc_report

QUESTION_FIELD: Final[str] = "question"
SAMPLING_FIELDS: Final[tuple[str, ...]] = (SAMPLE_N_FIELD, SAMPLE_TEMPERATURE_FIELD)
RESULTS_GLOB: Final[str] = "*.jsonl"
MIN_MODELS: Final[int] = 2


def _question_keys(rows: list[BenchmarkRow]) -> set[str]:
    return {str(row.get(ID_FIELD) or row[QUESTION_FIELD]) for row in rows}


def _model_summary(rows: list[BenchmarkRow], source: str) -> dict[str, Any]:
    report = auroc_report(rows, source)
    n_rows, n_hallucinations = report["n_rows"], report["n_hallucinations"]
    return {
        MODEL_ID_FIELD: report["run_config"][MODEL_ID_FIELD],
        "source": source,
        "n_rows": n_rows,
        "n_hallucinations": n_hallucinations,
        "accuracy": round(1 - n_hallucinations / n_rows, REPORT_DECIMALS),
        "detectors": report["detectors"],
        "sampling": {field: report["run_config"][field] for field in SAMPLING_FIELDS},
    }


def comparison_report(paths: list[Path]) -> dict[str, Any]:
    """One summary per model, after checking the runs are comparable."""
    if len(paths) < MIN_MODELS:
        raise RuntimeError(
            f"Comparing models needs at least {MIN_MODELS} results files; got {paths}."
        )

    loaded = [(path, load_rows(path)) for path in paths]
    first_path, first_rows = loaded[0]
    expected_questions = _question_keys(first_rows)

    summaries: list[dict[str, Any]] = []
    for path, rows in loaded:
        if _question_keys(rows) != expected_questions:
            raise RuntimeError(
                f"{path} and {first_path} cover different questions. Run every model over the "
                "same question set (run_benchmark --sweep does this)."
            )
        summaries.append(_model_summary(rows, str(path)))

    model_ids = [s[MODEL_ID_FIELD] for s in summaries]
    repeated = sorted({m for m in model_ids if model_ids.count(m) > 1})
    if repeated:
        raise RuntimeError(f"{repeated} appear more than once in {[str(p) for p in paths]}.")
    sampling = summaries[0]["sampling"]
    for summary in summaries[1:]:
        if summary["sampling"] != sampling:
            raise RuntimeError(
                f"{summary['source']} used sampling {summary['sampling']}, but "
                f"{summaries[0]['source']} used {sampling}. Compare runs made with the same "
                "SAMPLE_N and SAMPLE_TEMPERATURE."
            )

    return {"n_questions": len(expected_questions), "sampling": sampling, "models": summaries}


def _format_table(report: dict[str, Any]) -> str:
    detector_names = sorted({name for m in report["models"] for name in m["detectors"]})
    header = [
        "model",
        "scored",
        "wrong",
        "accuracy",
        *(f"{d} AUROC [95% CI]" for d in detector_names),
    ]
    lines = [header]
    for m in report["models"]:
        cells = [
            m[MODEL_ID_FIELD],
            str(m["n_rows"]),
            str(m["n_hallucinations"]),
            f"{m['accuracy']:.3f}",
        ]
        for name in detector_names:
            d = m["detectors"].get(name)
            cells.append(
                "n/a"
                if d is None
                else f"{d['auroc']:.3f} [{d['ci']['low']:.3f}, {d['ci']['high']:.3f}]"
            )
        lines.append(cells)
    widths = [max(len(row[i]) for row in lines) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) for row in lines
    )


def _results_paths(inputs: list[Path]) -> list[Path]:
    """Expand directories (a sweep's --out) into their results files."""
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            found = sorted(item.glob(RESULTS_GLOB))
            if not found:
                raise RuntimeError(f"{item} contains no {RESULTS_GLOB} results files.")
            paths.extend(found)
        else:
            paths.append(item)
    return paths


def main(inputs: list[Path], as_json: bool) -> None:
    report = comparison_report(_results_paths(inputs))
    if as_json:
        print(json.dumps(report, indent=2))
        return
    print(f"{report['n_questions']} questions, sampling {report['sampling']}\n")
    print(_format_table(report))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare models' accuracy and detector AUROCs.")
    p.add_argument(
        "--in",
        dest="inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Results files, or a directory written by run_benchmark --sweep.",
    )
    p.add_argument(
        "--json", dest="as_json", action="store_true", help="Print the full JSON report."
    )
    args = p.parse_args()
    main(args.inputs, args.as_json)
