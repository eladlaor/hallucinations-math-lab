"""hallucinations-math-lab demo CLI — reference demo of the response-evaluator library.

Subcommands:
- chat   : single-shot Q&A with the full detector + judge pipeline.
- detect : print the DetectionResult JSON for a single prompt (no judge call).

This is a demo. The library lives in `hallucinations_math_lab.*`; consumers who already own
their LLM should call `hallucinations_math_lab.evaluate_responses(...)` directly and skip
the demo's SLM provider plumbing entirely. See examples/ for that path.
"""

import json
import logging

import typer
from rich.console import Console

from hallucinations_math_lab import DetectorName, ResponseEvaluator
from hallucinations_math_lab.judges import GeminiJudge

app = typer.Typer(
    add_completion=False,
    help="hallucinations-math-lab — response evaluators for LLM-native projects.",
)
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
    """Run the full pipeline: evaluate -> (maybe) escalate to judge -> print final answer."""
    evaluator = ResponseEvaluator(judge=GeminiJudge())
    result = evaluator.evaluate(prompt)

    console.rule("Detection")
    console.print(_format_detector_scores(result.detector_scores))
    console.print(
        f"[bold]escalation_detector[/bold] = {result.escalation_detector}  "
        f"[bold]threshold[/bold] = {result.escalation_threshold:.3f}  "
        f"[bold]escalated[/bold] = {result.escalated}  "
        f"[bold]source[/bold] = {result.final_answer_source}"
    )
    console.rule("Final Answer")
    console.print(result.final_answer)
    if result.judge_reasoning:
        console.rule("Judge Reasoning")
        console.print(result.judge_reasoning)


@app.command()
def detect(prompt: str = typer.Argument(..., help="User prompt to inspect.")) -> None:
    """Run detection only (no judge call). Emits DetectionResult JSON."""
    evaluator = ResponseEvaluator(judge=None)
    result = evaluator.evaluate(prompt)
    console.print_json(json.dumps(result.model_dump(), default=str))


if __name__ == "__main__":
    app()
