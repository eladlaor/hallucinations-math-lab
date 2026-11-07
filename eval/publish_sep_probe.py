"""Publish a trained SEP probe and its calibration dataset to the Hugging Face Hub.

Inputs, all written by earlier steps of the release (SEP_CLASSIFIER.md, item 15):
  --probe-dir  the probe directory eval.train_sep_probe wrote;
  --dataset    the dataset directory eval.collect_sep_dataset wrote, which trained it;
  --heldout    eval.compute_auroc --json-out's report for an eval.run_benchmark run
               of this probe on questions disjoint from its training ones.

Nothing is uploaded unless the three agree: the held-out run scored this exact
probe (the SHA-256 of probe.safetensors and of probe_config.json both match),
on the probe's backbone, under a reference that is not a machine-specific
absolute path, and none of its scored question ids is a training id; the
dataset holds the probe's backbone, prompt and training question ids; and the
probe was fitted from committed code, so the commit the card links to is the
code that fitted it. The repo ids and the tag are checked first.

The model repo receives the probe directory with the held-out report merged
into metrics.json as `heldout_benchmark`, plus a model card (README.md)
rendered by eval.sep_model_card. The dataset repo receives the dataset
directory, uploaded first so the model repo never links to a dataset that has
not landed. Both uploads are tagged `--tag` once both have landed; a `--tag`
either repo already has is refused before anything is uploaded. If tagging the
model repo fails, the dataset repo's new tag is deleted, so a retry with the same
`--tag` is not refused.

The Hub token is the one `hf auth login` stored; it is never a command-line
argument, so it never lands in shell history or process listings.

Usage:
    uv run python -m eval.publish_sep_probe --probe-dir DIR --dataset DIR
        --heldout auroc.json --repo OWNER/NAME --dataset-repo OWNER/NAME --tag v1
        [--dry-run]

`--dry-run` runs every check, renders the card and prints it, and uploads nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from huggingface_hub import CommitInfo, HfApi
from pydantic import ValidationError

from eval.sep_dataset_io import SEPDataset, load_sep_dataset
from eval.sep_model_card import HeldoutReport, PublishTarget, render_model_card
from eval.sep_probe_provenance import MAX_IDS_IN_ERROR
from hallucinations_math_lab.constants import (
    LOG_EXTRA_KEY,
    SEP_DATASET_DIR_FILES,
    SEP_METRICS_FILE,
    SEP_PROBE_CONFIG_FILE,
    SEP_PROBE_DIR_FILES,
    SEP_PROBE_WEIGHTS_FILE,
    SEP_SERVED_TOKEN_POSITION,
    SEP_TRAIN_IDS_FILE,
    SEP_TRAINING_LOG_FILE,
    DetectorName,
)
from hallucinations_math_lab.custom_types.sep_probe_types import (
    SEPProbeMetrics,
    SEPPublishedMetrics,
    SEPTrainIds,
)
from hallucinations_math_lab.services.semantic_entropy_probe import (
    LoadedSEPProbe,
    load_probe_dir,
    validate_hub_repo_id,
)

logger = logging.getLogger("publish_sep_probe")

# The Hub renders a repo's README.md as its model card.
MODEL_CARD_FILE: Final[str] = "README.md"
MODEL_REPO_TYPE: Final[str] = "model"
DATASET_REPO_TYPE: Final[str] = "dataset"
_JSON_INDENT: Final[int] = 2
# A tag both git and an hf:// reference accept: letters, digits, '.', '_' and '-',
# starting and ending with a letter or digit, with no '..' and no '.lock' ending.
TAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?!.*\.\.)(?!.*\.lock$)[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
)


def load_heldout(path: Path) -> tuple[dict[str, Any], HeldoutReport]:
    """compute_auroc's report at `path`: verbatim, and as the checked view."""
    try:
        raw = json.loads(path.read_text())
        return raw, HeldoutReport.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as e:
        raise ValueError(
            f"Cannot read the held-out benchmark report {path} (eval.compute_auroc "
            f"--json-out): {e!s}"
        ) from e


def _load_metrics(probe_dir: Path) -> SEPProbeMetrics:
    path = probe_dir / SEP_METRICS_FILE
    try:
        return SEPProbeMetrics.model_validate_json(path.read_text())
    except (OSError, ValidationError) as e:
        raise ValueError(
            f"Cannot read {path} as the trainer's metrics (a probe directory is published "
            f"once, from eval.train_sep_probe's output): {e!s}"
        ) from e


def check_target(target: PublishTarget) -> None:
    """Raise unless both repo ids are `<owner>/<name>` Hub repo ids and the tag is a valid
    tag name, so a bad one fails before anything is written or uploaded."""
    for flag, repo_id in (("--repo", target.model_repo), ("--dataset-repo", target.dataset_repo)):
        try:
            validate_hub_repo_id(repo_id)
        except ValueError as e:
            raise ValueError(f"{flag}: {e!s}") from e
    if not TAG_PATTERN.fullmatch(target.tag):
        raise ValueError(
            f"--tag {target.tag!r} is not a valid tag: use letters, digits, '.', '_' and '-', "
            "starting and ending with a letter or digit, e.g. v1 or v1.0."
        )


def check_probe(probe: LoadedSEPProbe, probe_dir: Path, metrics: SEPProbeMetrics) -> None:
    """Raise unless the probe directory is complete, fitted from committed code, and its
    metrics describe the window it serves."""
    missing = [name for name in SEP_PROBE_DIR_FILES if not (probe_dir / name).is_file()]
    if missing:
        raise ValueError(f"SEP probe directory {probe_dir} lacks {missing}; retrain it.")
    training = probe.config.training
    if training.repo_dirty:
        raise ValueError(
            f"SEP probe {probe_dir} was fitted from uncommitted code (repo_dirty at commit "
            f"{training.repo_commit}), so no commit names the code that fitted it. Commit, "
            "refit with eval.train_sep_probe, and benchmark the refitted probe."
        )
    window = metrics.selection[SEP_SERVED_TOKEN_POSITION].window
    if window != probe.config.features.layers:
        raise ValueError(
            f"{probe_dir / SEP_METRICS_FILE} reports the {SEP_SERVED_TOKEN_POSITION} window "
            f"{list(window)}, but the probe reads layers {list(probe.config.features.layers)}."
        )


def check_heldout(heldout: HeldoutReport, probe: LoadedSEPProbe, probe_dir: Path) -> None:
    """Raise unless the held-out run scored this exact probe, with SEP, on its backbone,
    and on no question the probe was trained on."""
    sep = DetectorName.SEMANTIC_ENTROPY_PROBE
    scored = heldout.run_config.sep_probe
    if scored is None or sep not in heldout.detectors:
        raise ValueError(
            f"The held-out benchmark has no {sep} AUROC (detectors: "
            f"{sorted(str(d) for d in heldout.detectors)}). Run eval.run_benchmark with SEP "
            f"enabled and this probe, on a disjoint slice, then eval.compute_auroc --json-out."
        )
    expected = {
        "weights_sha256": (scored.weights_sha256, probe.weights_sha256),
        "config_sha256": (scored.config_sha256, probe.config_sha256),
    }
    for field, (benchmarked, published) in expected.items():
        if benchmarked != published:
            raise ValueError(
                f"The held-out benchmark scored a different SEP probe: its {field} is "
                f"{benchmarked}, but {probe_dir} has {published}. Benchmark the probe being "
                "published."
            )
    if Path(scored.ref).is_absolute():
        raise ValueError(
            f"The held-out benchmark records the probe at the absolute path {scored.ref!r}, "
            "which would publish a machine-specific location. Re-run it with a relative "
            "SEP_PROBE_PATH (or the data/probes/<slug> default) or an hf:// reference."
        )
    model_id = probe.config.backbone.model_id
    if heldout.run_config.model_id != model_id:
        raise ValueError(
            f"The held-out benchmark ran {heldout.run_config.model_id}, but the probe was "
            f"trained on {model_id}."
        )
    if probe.train_ids is None:
        raise ValueError(f"SEP probe {probe_dir} has no training ids to check the held-out set.")
    overlap = sorted(set(heldout.question_ids) & set(probe.train_ids.question_ids))
    if overlap:
        raise ValueError(
            f"{len(overlap)} held-out benchmark questions trained the SEP probe {probe_dir} "
            f"(e.g. {overlap[:MAX_IDS_IN_ERROR]}), so its AUROC is not held-out. Benchmark it "
            "on a slice disjoint from its calibration set."
        )


def check_dataset(
    dataset: SEPDataset, dataset_dir: Path, probe: LoadedSEPProbe, probe_dir: Path
) -> None:
    """Raise unless `dataset_dir` is complete and is the dataset that trained `probe`: same
    backbone, prompt, calibration set, question ids in training order, and training log."""
    missing = [name for name in SEP_DATASET_DIR_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise ValueError(f"SEP dataset directory {dataset_dir} lacks {missing}.")
    try:
        same_log = (dataset_dir / SEP_TRAINING_LOG_FILE).read_bytes() == (
            probe_dir / SEP_TRAINING_LOG_FILE
        ).read_bytes()
    except OSError as e:
        raise RuntimeError(f"Cannot compare the {SEP_TRAINING_LOG_FILE} files: {e!s}") from e
    if not same_log:
        raise ValueError(
            f"{probe_dir / SEP_TRAINING_LOG_FILE} is not {dataset_dir / SEP_TRAINING_LOG_FILE}; "
            "the trainer copies the dataset's log, so this dataset did not train this probe."
        )
    config = probe.config
    collected = dataset.config
    pairs: dict[str, tuple[object, object]] = {
        "backbone": (collected.backbone, config.backbone),
        "prompt": (collected.prompt, config.prompt),
        "calibration_set_sha256": (
            collected.calibration_set_sha256,
            config.training.calibration_set_sha256,
        ),
    }
    for field, (in_dataset, in_probe) in pairs.items():
        if in_dataset != in_probe:
            raise ValueError(
                f"SEP dataset {dataset_dir} did not train this probe: its {field} is "
                f"{in_dataset}, the probe's is {in_probe}."
            )
    if probe.train_ids is None:
        raise ValueError("The SEP probe has no training ids to compare with the dataset.")
    dataset_ids = tuple(row.id for row in dataset.rows)
    if dataset_ids != probe.train_ids.question_ids:
        raise ValueError(
            f"SEP dataset {dataset_dir} did not train this probe: its {len(dataset_ids)} "
            f"question ids differ from the probe's {len(probe.train_ids.question_ids)} "
            "training question ids."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_staged(
    staging_dir: Path, probe: LoadedSEPProbe, probe_dir: Path, dataset_dir: Path
) -> None:
    """Raise unless every staged probe file is what was checked, even if `probe_dir` changed
    since: the benchmarked weights and config, the training ids compared with the dataset,
    and the dataset's training log."""
    try:
        staged = {
            SEP_PROBE_WEIGHTS_FILE: _sha256(staging_dir / SEP_PROBE_WEIGHTS_FILE)
            == probe.weights_sha256,
            SEP_PROBE_CONFIG_FILE: _sha256(staging_dir / SEP_PROBE_CONFIG_FILE)
            == probe.config_sha256,
            SEP_TRAIN_IDS_FILE: SEPTrainIds.model_validate_json(
                (staging_dir / SEP_TRAIN_IDS_FILE).read_text()
            )
            == probe.train_ids,
            SEP_TRAINING_LOG_FILE: (staging_dir / SEP_TRAINING_LOG_FILE).read_bytes()
            == (dataset_dir / SEP_TRAINING_LOG_FILE).read_bytes(),
        }
    except (OSError, ValidationError) as e:
        raise RuntimeError(f"Cannot verify the staged release in {staging_dir}: {e!s}") from e
    changed = [name for name, same in staged.items() if not same]
    if changed:
        raise RuntimeError(
            f"{changed} in {probe_dir} or {dataset_dir} changed while the release was staged; "
            "re-run the publish."
        )


def stage_release(
    probe_dir: Path, dataset_dir: Path, heldout_path: Path, target: PublishTarget, staging_dir: Path
) -> str:
    """Check the three inputs agree, then write the model repo's folder to `staging_dir`
    (absent or empty): the probe's files, metrics.json with the held-out report merged in,
    and the model card, which is returned. The inputs are checked before anything is written."""
    check_target(target)
    probe = load_probe_dir(probe_dir)
    metrics = _load_metrics(probe_dir)
    check_probe(probe, probe_dir, metrics)
    raw_heldout, heldout = load_heldout(heldout_path)
    check_heldout(heldout, probe, probe_dir)
    check_dataset(load_sep_dataset(dataset_dir), dataset_dir, probe, probe_dir)

    published = SEPPublishedMetrics(**metrics.model_dump(), heldout_benchmark=raw_heldout)
    card = render_model_card(probe.config, published, heldout, target)
    if staging_dir.exists() and any(staging_dir.iterdir()):
        raise FileExistsError(f"Staging directory {staging_dir} is not empty.")
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
        for name in SEP_PROBE_DIR_FILES:
            if name != SEP_METRICS_FILE:
                shutil.copyfile(probe_dir / name, staging_dir / name)
        (staging_dir / SEP_METRICS_FILE).write_text(
            published.model_dump_json(indent=_JSON_INDENT) + "\n"
        )
        (staging_dir / MODEL_CARD_FILE).write_text(card)
    except OSError as e:
        raise RuntimeError(f"Cannot stage the SEP probe release in {staging_dir}: {e!s}") from e
    verify_staged(staging_dir, probe, probe_dir, dataset_dir)
    logger.info(
        "Staged the SEP probe release.",
        extra={
            LOG_EXTRA_KEY: {
                "probe_dir": str(probe_dir),
                "dataset_dir": str(dataset_dir),
                "staging_dir": str(staging_dir),
                "model_repo": target.model_repo,
                "dataset_repo": target.dataset_repo,
                "tag": target.tag,
                "heldout_sep_auroc": heldout.detectors[DetectorName.SEMANTIC_ENTROPY_PROBE].auroc,
            }
        },
    )
    return card


def _hub_error(action: str, repo_id: str, repo_type: str, error: OSError) -> RuntimeError:
    """Log a failed Hub call and return the error to raise, naming the repo."""
    logger.error(
        "Hub call failed.",
        extra={
            LOG_EXTRA_KEY: {
                "action": action,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "error": repr(error),
            }
        },
    )
    return RuntimeError(f"Cannot {action} {repo_type} repo {repo_id} on the Hub: {error!s}")


def _check_tag_free(api: HfApi, repo_id: str, repo_type: str, tag: str) -> None:
    """Raise if `repo_id` exists and already has `tag`."""
    try:
        tags = (
            {ref.name for ref in api.list_repo_refs(repo_id, repo_type=repo_type).tags}
            if api.repo_exists(repo_id, repo_type=repo_type)
            else set()
        )
    except OSError as e:  # HfHubHTTPError is an OSError
        raise _hub_error("list the tags of", repo_id, repo_type, e) from e
    if tag in tags:
        raise ValueError(
            f"{repo_type.capitalize()} repo {repo_id} already has tag {tag!r}; publish under a "
            "new --tag. Nothing was uploaded."
        )


def _upload(
    api: HfApi, repo_id: str, repo_type: str, folder: Path, files: Sequence[str], message: str
) -> str:
    """Create the repo if needed and upload `files` from `folder`; returns the commit sha."""
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        info = api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=folder,
            allow_patterns=list(files),
            commit_message=message,
        )
    except OSError as e:  # HfHubHTTPError is an OSError
        raise _hub_error("upload to", repo_id, repo_type, e) from e
    if not isinstance(info, CommitInfo):
        raise TypeError(f"upload_folder returned {type(info).__name__} for {repo_id}.")
    logger.info(
        "Uploaded to the Hub.",
        extra={LOG_EXTRA_KEY: {"repo_id": repo_id, "repo_type": repo_type, "commit": info.oid}},
    )
    return info.oid


def _tag_all(api: HfApi, commits: dict[tuple[str, str], str], tag: str) -> None:
    """Tag every commit, in order. If one fails, delete the tags already made, so either
    every repo has `tag` or none does, and a retry with the same tag is not refused."""
    tagged: list[tuple[str, str]] = []
    for (repo_id, repo_type), commit in commits.items():
        try:
            api.create_tag(repo_id, tag=tag, revision=commit, repo_type=repo_type)
        except OSError as e:  # HfHubHTTPError is an OSError
            error = _hub_error(f"tag {tag} in", repo_id, repo_type, e)
            for done_id, done_type in tagged:
                try:
                    api.delete_tag(done_id, tag=tag, repo_type=done_type)
                except OSError as undo:
                    cleanup = _hub_error(f"delete tag {tag} from", done_id, done_type, undo)
                    raise RuntimeError(
                        f"{error} Rolling back also failed: {cleanup} Delete tag {tag} from "
                        f"{done_type} repo {done_id} on the Hub before retrying with --tag {tag}."
                    ) from undo
            raise error from e
        tagged.append((repo_id, repo_type))


def upload_release(api: HfApi, staging_dir: Path, dataset_dir: Path, target: PublishTarget) -> None:
    """Refuse a tag either repo already has, then upload the dataset, then the staged model
    folder, then tag both commits. The dataset goes first so the model repo, the one users
    load, never links to a dataset that has not landed."""
    repos = ((target.dataset_repo, DATASET_REPO_TYPE), (target.model_repo, MODEL_REPO_TYPE))
    for repo_id, repo_type in repos:
        _check_tag_free(api, repo_id, repo_type, target.tag)
    commits = {
        (target.dataset_repo, DATASET_REPO_TYPE): _upload(
            api,
            target.dataset_repo,
            DATASET_REPO_TYPE,
            dataset_dir,
            SEP_DATASET_DIR_FILES,
            f"Release SEP calibration dataset {target.tag}",
        ),
        (target.model_repo, MODEL_REPO_TYPE): _upload(
            api,
            target.model_repo,
            MODEL_REPO_TYPE,
            staging_dir,
            (*SEP_PROBE_DIR_FILES, MODEL_CARD_FILE),
            f"Release SEP probe {target.tag}",
        ),
    }
    _tag_all(api, commits, target.tag)
    logger.info(
        "Published the SEP probe.",
        extra={LOG_EXTRA_KEY: {"probe_uri": target.probe_uri, "dataset_repo": target.dataset_repo}},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish a SEP probe and its dataset to the Hub (token: `hf auth login`)."
    )
    parser.add_argument(
        "--probe-dir", type=Path, required=True, help="eval.train_sep_probe output."
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="The dataset directory that trained it."
    )
    parser.add_argument(
        "--heldout", type=Path, required=True, help="eval.compute_auroc --json-out report."
    )
    parser.add_argument("--repo", required=True, help="Model repo, <owner>/<name>.")
    parser.add_argument("--dataset-repo", required=True, help="Dataset repo, <owner>/<name>.")
    parser.add_argument("--tag", required=True, help="Tag for both commits, e.g. v1.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Check and print the card; upload nothing."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = PublishTarget(model_repo=args.repo, dataset_repo=args.dataset_repo, tag=args.tag)
    with tempfile.TemporaryDirectory() as staging:
        card = stage_release(args.probe_dir, args.dataset, args.heldout, target, Path(staging))
        if args.dry_run:
            print(card)
            logger.info(
                "Dry run: nothing uploaded.", extra={LOG_EXTRA_KEY: {"probe_uri": target.probe_uri}}
            )
            return 0
        upload_release(HfApi(), Path(staging), args.dataset, target)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.exit(main())
