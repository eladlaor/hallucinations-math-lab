"""Immutable values for the web playground: routes, network binding, filesystem locations."""

from enum import StrEnum
from pathlib import Path
from typing import Final

from hallucinations_math_lab.constants import DetectorName

# Network: localhost only, never a public interface.
DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8765

# Routes.
API_PREFIX: Final[str] = "/api"
ROUTE_STATUS: Final[str] = "/status"
ROUTE_PRESETS: Final[str] = "/presets"
ROUTE_SCORE: Final[str] = "/score"
ROUTE_ASK: Final[str] = "/ask"
ROUTE_RESULTS: Final[str] = "/results"
ROUTE_RESULTS_REPORT: Final[str] = "/results/report"
STATIC_MOUNT_PATH: Final[str] = "/"
STATIC_MOUNT_NAME: Final[str] = "ui"

# Filesystem. The repository root is two levels above this package (demo/web/).
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
STATIC_DIR: Final[Path] = Path(__file__).resolve().parent / "static"
INDEX_HTML: Final[str] = "index.html"
RESULTS_DIR: Final[Path] = REPO_ROOT / "eval" / "results"

# The example whose SCENARIOS are the Score tab's presets (imported, never copied).
PRESETS_MODULE: Final[str] = "examples.score_precollected_samples"
PRESETS_ATTRIBUTE: Final[str] = "SCENARIOS"

# The detectors that can return an SE trace (`se_trace`): both SE estimators.
SE_TRACE_DETECTORS: Final[frozenset[DetectorName]] = frozenset(
    {DetectorName.SEMANTIC_ENTROPY, DetectorName.DISCRETE_SEMANTIC_ENTROPY}
)


class AskMode(StrEnum):
    """The two `demo.cli` commands the Ask tab mirrors."""

    DETECT = "detect"
    CHAT = "chat"


class ResultsEntryKind(StrEnum):
    """A results file from one run, or a `run_benchmark --sweep` directory."""

    RUN = "run"
    SWEEP = "sweep"
