"""Typer CLI entrypoint. Subcommands:
- chat   : interactive single-shot Q&A with full detector+judge pipeline
- detect : print the DetectionResult JSON for a single prompt (no judge escalation)
"""

import json
import logging

import typer
from rich.console import Console

from taste.agents.detector_agent import DetectorAgent
from taste.agents.judge_agent import JudgeAgent
from taste.constants import DetectorName

app = typer.Typer(add_completion=False, help="Semantic-Entropy hallucination detector CLI.")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


_DETECTOR_LABELS: dict[DetectorName, str] = {
    DetectorName.PREDICTIVE_ENTROPY: "PE",
    DetectorName.SEMANTIC_ENTROPY_PROBE: "SEP",
    DetectorName.SEMANTIC_ENTROPY: "SE",
}


def _format_detector_scores(scores: dict[DetectorName, float]) -> str:
    """Render the detector_scores dict in canonical PE | SEP | SE order."""
    parts: list[str] = []
    for name in (
        DetectorName.PREDICTIVE_ENTROPY,
        DetectorName.SEMANTIC_ENTROPY_PROBE,
        DetectorName.SEMANTIC_ENTROPY,
    ):
        if name in scores:
            parts.append(f"[bold]{_DETECTOR_LABELS[name]}[/bold] = {scores[name]:.3f}")
    return "  ".join(parts) if parts else "[dim]no detectors ran[/dim]"


@app.command()
def chat(prompt: str = typer.Argument(..., help="User prompt to answer.")) -> None:
    """Run the full pipeline: detect -> (maybe) escalate to judge -> print final answer."""
    detector = DetectorAgent()
    judge = JudgeAgent()
    detection = detector.detect(prompt)
    final = judge.maybe_escalate(detection)

    console.rule("Detection")
    console.print(_format_detector_scores(final.detector_scores))
    console.print(
        f"[bold]escalation_detector[/bold] = {final.escalation_detector}  "
        f"[bold]threshold[/bold] = {final.escalation_threshold:.3f}  "
        f"[bold]escalated[/bold] = {final.escalated}  "
        f"[bold]source[/bold] = {final.final_answer_source}"
    )
    console.rule("Final Answer")
    console.print(final.final_answer)
    if final.judge_reasoning:
        console.rule("Judge Reasoning")
        console.print(final.judge_reasoning)


@app.command()
def detect(prompt: str = typer.Argument(..., help="User prompt to inspect.")) -> None:
    """Run detection only (no judge call). Emits DetectionResult JSON."""
    detector = DetectorAgent()
    result = detector.detect(prompt)
    console.print_json(json.dumps(result.model_dump(), default=str))


if __name__ == "__main__":
    app()
