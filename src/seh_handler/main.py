"""Typer CLI entrypoint. Subcommands:
- chat   : interactive single-shot Q&A with full detector+judge pipeline
- detect : print the DetectionResult JSON for a single prompt (no judge escalation)
"""

import json
import logging

import typer
from rich.console import Console

from seh_handler.agents.detector_agent import DetectorAgent
from seh_handler.agents.judge_agent import JudgeAgent

app = typer.Typer(add_completion=False, help="Semantic-Entropy hallucination detector CLI.")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@app.command()
def chat(prompt: str = typer.Argument(..., help="User prompt to answer.")) -> None:
    """Run the full pipeline: detect -> (maybe) escalate to judge -> print final answer."""
    detector = DetectorAgent()
    judge = JudgeAgent()
    detection = detector.detect(prompt)
    final = judge.maybe_escalate(detection)

    console.rule("Detection")
    console.print(
        f"[bold]SE[/bold] = {final.semantic_entropy:.3f} nats  "
        f"[bold]PE[/bold] = {final.predictive_entropy:.3f}  "
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
