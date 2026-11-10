"""Benchmark results on disk, for the Measure the detectors tab: listing them, and reporting
on one without leaving the results dir.

The reports themselves come from eval.compute_auroc and eval.compare_models
unchanged; this module only finds the files and guards the path.
"""

import logging
from pathlib import Path

from demo.web.api_schemas import ResultsEntry, ResultsListing, ResultsReport
from demo.web.constants import ResultsEntryKind
from eval.benchmark_results import load_rows
from eval.compare_models import RESULTS_GLOB, comparison_report
from eval.compute_auroc import auroc_report
from hallucinations_math_lab.constants import LOG_EXTRA_KEY

logger = logging.getLogger(__name__)


class ResultsPathError(ValueError):
    """A requested path that points outside the results directory."""


def list_results(results_dir: Path) -> ResultsListing:
    """Top-level results files (single runs) and directories of them (sweeps)."""
    entries: list[ResultsEntry] = []
    if results_dir.is_dir():
        for item in sorted(results_dir.iterdir()):
            if item.is_dir():
                n_files = len(list(item.glob(RESULTS_GLOB)))
                if n_files:
                    entries.append(
                        ResultsEntry(path=item.name, kind=ResultsEntryKind.SWEEP, n_files=n_files)
                    )
            elif item.match(RESULTS_GLOB):
                entries.append(ResultsEntry(path=item.name, kind=ResultsEntryKind.RUN, n_files=1))
    logger.info(
        "Listed benchmark results.",
        extra={LOG_EXTRA_KEY: {"results_dir": str(results_dir), "n_entries": len(entries)}},
    )
    return ResultsListing(results_dir=str(results_dir), entries=entries)


def resolve_results_path(results_dir: Path, relative: str) -> Path:
    """`relative` under `results_dir`, after resolving `..` and symlinks.

    Raises ResultsPathError when it lands outside, FileNotFoundError when it
    does not exist.
    """
    root = results_dir.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ResultsPathError(f"{relative!r} is outside the results directory {results_dir}.")
    if not target.exists():
        raise FileNotFoundError(f"{relative!r} does not exist in {results_dir}.")
    return target


def build_report(results_dir: Path, relative: str) -> ResultsReport:
    """AUROC report for a results file, model comparison for a sweep directory.

    Raises ResultsPathError / FileNotFoundError for a bad path, RuntimeError
    (from the eval scripts) for a file that cannot be scored.
    """
    target = resolve_results_path(results_dir, relative)
    try:
        if target.is_dir():
            paths = sorted(target.glob(RESULTS_GLOB))
            if not paths:
                raise RuntimeError(f"{relative} contains no {RESULTS_GLOB} results files.")
            return ResultsReport(
                path=relative, kind=ResultsEntryKind.SWEEP, report=comparison_report(paths)
            )
        return ResultsReport(
            path=relative,
            kind=ResultsEntryKind.RUN,
            report=auroc_report(load_rows(target), source=relative),
        )
    except RuntimeError as e:
        logger.error(
            "Could not report on benchmark results.",
            extra={LOG_EXTRA_KEY: {"path": relative, "error": str(e)}},
        )
        raise
