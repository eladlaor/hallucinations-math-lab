"""The license terms on both Hub cards, and the dataset repo's card.

Contract pinned here:
  - Both cards carry the backbone's Hub license (`license:` front matter), and the
    model card links the backbone's own license page (`license_link:`).
  - A backbone with no recorded license renders no card, so nothing is published
    under unstated terms.
  - The dataset card is rendered purely from the probe's config and where the
    release goes; a golden file pins its text. It names every rows.parquet column,
    prints the probe's own refit command, and attributes the questions to TriviaQA.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.datasets.load_triviaqa import DATASET_NAME
from eval.sep_dataset_io import ROWS_SCHEMA
from eval.sep_model_card import (
    BACKBONE_LICENSES,
    GEMMA_4_LICENSE_URL,
    PublishTarget,
    backbone_license,
    refit_command,
    render_dataset_card,
)
from hallucinations_math_lab.custom_types.sep_probe_types import SEPProbeConfig
from tests.unit.sep_probe_fixtures import probe_config

_GOLDEN_DATASET_CARD = Path(__file__).parent / "fixtures" / "sep_dataset_card.golden.md"
_MODEL_ID = "google/gemma-4-E2B-it"
_TARGET = PublishTarget(
    model_repo="owner/sep-probe-model", dataset_repo="owner/sep-calibration-model", tag="v1"
)


def _config(model_id: str = _MODEL_ID) -> SEPProbeConfig:
    return probe_config(layers=(1, 2), model_id=model_id)


def _front_matter(card: str) -> list[str]:
    _, front, _ = card.split("---\n", 2)
    return front.splitlines()


def test_dataset_card_snapshot() -> None:
    assert render_dataset_card(_config(), _TARGET) == _GOLDEN_DATASET_CARD.read_text()


def test_dataset_card_front_matter_carries_the_license_and_source() -> None:
    front = _front_matter(render_dataset_card(_config(), _TARGET))
    assert "license: apache-2.0" in front
    assert f"- {DATASET_NAME}" in front
    assert f"- {_TARGET.model_repo}" in front


def test_dataset_card_names_every_row_column() -> None:
    card = render_dataset_card(_config(), _TARGET)
    missing = [name for name in ROWS_SCHEMA.names if f"`{name}`" not in card]
    assert not missing


def test_dataset_card_prints_the_probe_refit_command() -> None:
    config = _config()
    assert refit_command(config) in render_dataset_card(config, _TARGET)


def test_dataset_card_attributes_the_questions_to_triviaqa() -> None:
    card = render_dataset_card(_config(), _TARGET)
    assert f"https://huggingface.co/datasets/{DATASET_NAME}" in card
    assert "Joshi" in card
    assert "unknown" in card


def test_gemma_4_backbones_are_apache_2_0_with_the_gemma_4_license_page() -> None:
    for model_id in (
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-31B-it",
    ):
        terms = backbone_license(model_id)
        assert terms.license.hub_id == "apache-2.0"
        assert terms.terms_url == GEMMA_4_LICENSE_URL


def test_backbone_without_a_recorded_license_renders_no_card() -> None:
    assert "org/model" not in BACKBONE_LICENSES
    with pytest.raises(ValueError, match="org/model"):
        render_dataset_card(_config("org/model"), _TARGET)


def test_dataset_card_states_the_license_scope_before_the_first_paragraph() -> None:
    _, _, body = render_dataset_card(_config(), _TARGET).split("---\n", 2)
    first_paragraph = body.strip().split("\n\n")[1]
    assert first_paragraph.startswith("**License:**")
    assert "TriviaQA" in first_paragraph


def test_dataset_card_names_the_fitting_commit_not_a_collection_commit() -> None:
    config = _config()
    card = render_dataset_card(config, _TARGET)
    assert "fitted with [hallucinations-math-lab]" in card
    assert "the commit that collected the data is not recorded" in card.lower()
    assert "built the data" not in card
