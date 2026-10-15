"""Score the benchmark output with Vertex Gen AI Evaluation Service.

Computes:
  - Groundedness on the final-answer column (escalated mixture)
  - Pairwise: small greedy vs final (judge-augmented) — Gemini judge

Run AFTER run_benchmark.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(path: Path) -> None:
    try:
        import pandas as pd
        import vertexai
        from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples
    except ImportError as e:
        raise RuntimeError(
            "vertex_genai_eval requires google-cloud-aiplatform[evaluation]. "
            f"Install missing dep: {e!s}"
        ) from e

    from hallucinations_math_lab.config import get_settings

    settings = get_settings()
    if not settings.gcp_project_id:
        raise RuntimeError("GCP_PROJECT_ID is empty; set it in .env before running.")
    vertexai.init(project=settings.gcp_project_id, location=settings.gcp_region)

    rows = [json.loads(line) for line in path.open()]
    df = pd.DataFrame(
        [
            {
                "prompt": r["question"],
                "response": r["final_answer"],
                "baseline_model_response": r["greedy_answer"],
            }
            for r in rows
        ]
    )

    task = EvalTask(
        dataset=df,
        metrics=[
            MetricPromptTemplateExamples.Pointwise.GROUNDEDNESS,
            MetricPromptTemplateExamples.Pairwise.QUESTION_ANSWERING_QUALITY,
        ],
    )
    result = task.evaluate()
    print(result.summary_metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="path", type=Path, required=True)
    args = p.parse_args()
    main(args.path)
