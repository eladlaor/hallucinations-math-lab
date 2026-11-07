"""Which commit of this repository produced an artifact, and whether it had local changes.

A probe records both, so a published probe names the exact code that fitted it,
and a probe fitted from uncommitted code says so.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from hallucinations_math_lab.constants import GIT_COMMIT_SHA_PATTERN

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
_GIT: Final[str] = "git"
_HEAD_COMMAND: Final[tuple[str, ...]] = ("rev-parse", "HEAD")
# Porcelain status prints one line per changed or untracked file, nothing when clean.
_STATUS_COMMAND: Final[tuple[str, ...]] = ("status", "--porcelain")


@dataclass(frozen=True)
class RepoState:
    commit: str
    dirty: bool


def _git(repo_root: Path, *args: str) -> str:
    """stdout of `git -C repo_root args`; raises RuntimeError naming the command on failure."""
    command = [_GIT, "-C", str(repo_root), *args]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError) as e:
        stderr = getattr(e, "stderr", None) or ""
        raise RuntimeError(
            f"Cannot read the repository state: `{' '.join(command)}` failed: {e!s} "
            f"{stderr.strip()}".strip()
        ) from e
    return result.stdout


def read_repo_state(repo_root: Path = REPO_ROOT) -> RepoState:
    """HEAD's commit sha and whether the working tree differs from it (untracked files count)."""
    commit = _git(repo_root, *_HEAD_COMMAND).strip()
    if not re.fullmatch(GIT_COMMIT_SHA_PATTERN, commit):
        raise RuntimeError(f"git reported HEAD {commit!r} in {repo_root}, not a full commit sha.")
    return RepoState(commit=commit, dirty=bool(_git(repo_root, *_STATUS_COMMAND).strip()))
