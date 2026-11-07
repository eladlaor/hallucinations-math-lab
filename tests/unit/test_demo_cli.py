"""The demo CLI prints each detector score on its own scale."""

from demo.cli import _format_detector_scores, _format_score
from hallucinations_math_lab import DetectorName


def test_sep_score_is_a_percentage_labelled_p_high_se() -> None:
    line = _format_detector_scores(
        {
            DetectorName.PREDICTIVE_ENTROPY: 0.4,
            DetectorName.SEMANTIC_ENTROPY_PROBE: 0.7234,
            DetectorName.SEMANTIC_ENTROPY: 1.8343,
        }
    )

    assert "[bold]SEP P(high SE)[/bold] = 72.3%" in line
    assert "[bold]PE[/bold] = 0.400" in line
    assert "[bold]SE[/bold] = 1.834" in line
    assert "nats" not in line


def test_sep_threshold_value_is_a_percentage() -> None:
    assert _format_score(DetectorName.SEMANTIC_ENTROPY_PROBE, 0.5) == "50.0%"


def test_nats_threshold_value_is_a_plain_number() -> None:
    assert _format_score(DetectorName.SEMANTIC_ENTROPY, 1.0) == "1.000"
