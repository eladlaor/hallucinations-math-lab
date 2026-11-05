"""The default TriviaQA question sets, built on first use and cached under data/.

The benchmark and SEP probe training read these when no question file is
given, so a new user brings no data. Each is a fixed slice of the seeded
shuffle in load_triviaqa.py, so every machine builds the same questions. The
two slices do not overlap, which keeps the probe's training questions out of
the benchmark.

The questions are downloaded (about 7 MB, once) rather than committed:
TriviaQA's redistribution terms are unclear, and data/ is gitignored.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from hallucinations_math_lab.constants import LOG_EXTRA_KEY

logger = logging.getLogger(__name__)

QUESTIONS_DIR: Final[Path] = Path("data/questions")

# (n, offset, out) -> rows; load_triviaqa.load_triviaqa_subset matches it.
QuestionLoader = Callable[..., list[dict]]


@dataclass(frozen=True)
class QuestionSlice:
    name: str
    offset: int
    n: int
    directory: Path = QUESTIONS_DIR

    @property
    def path(self) -> Path:
        return self.directory / f"triviaqa_{self.name}_{self.offset}-{self.offset + self.n}.jsonl"


CALIBRATION_SLICE: Final[QuestionSlice] = QuestionSlice(name="calibration", offset=0, n=500)
BENCHMARK_SLICE: Final[QuestionSlice] = QuestionSlice(name="benchmark", offset=500, n=200)


def _download_loader(n: int, offset: int, out: Path) -> list[dict]:
    # Imported lazily: it pulls in the `datasets` package, needed only on first build.
    from eval.datasets.load_triviaqa import load_triviaqa_subset

    return load_triviaqa_subset(n=n, offset=offset, out=out)


def ensure_question_set(s: QuestionSlice, loader: QuestionLoader | None = None) -> Path:
    """Path to the slice's JSONL, building it first if it is not on disk yet."""
    if s.path.exists():
        return s.path
    load = loader or _download_loader
    ctx = {"slice": s.name, "offset": s.offset, "n": s.n, "path": str(s.path)}
    logger.info("Building default question set", extra={LOG_EXTRA_KEY: ctx})
    try:
        load(n=s.n, offset=s.offset, out=s.path)
    except Exception as e:
        s.path.unlink(missing_ok=True)
        logger.error("Default question set build failed", extra={LOG_EXTRA_KEY: ctx})
        raise RuntimeError(
            f"Could not build the {s.name} question set [{s.offset}, {s.offset + s.n}) "
            f"at {s.path}: {e!s}. It downloads TriviaQA from Hugging Face; check the "
            "network, or pass your own question file."
        ) from e
    if not s.path.exists():
        raise RuntimeError(f"Building the {s.name} question set wrote nothing to {s.path}.")
    return s.path
