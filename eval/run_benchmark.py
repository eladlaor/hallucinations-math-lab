"""Run the benchmark over TriviaQA-subset + adversarial prompts.

Outputs a JSONL of per-question rows containing greedy/judge answers, SE, PE, escalation.
Downstream: compute_auroc.py, vertex_genai_eval.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from seh_handler.agents.detector_agent import DetectorAgent
from seh_handler.agents.judge_agent import JudgeAgent
from seh_handler.constants import LOG_EXTRA_KEY

logger = logging.getLogger(__name__)


def run(in_path: Path, out_path: Path, judge: bool = True, limit: int | None = None) -> None:
    detector = DetectorAgent()
    judge_agent = JudgeAgent() if judge else None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open() as f_in, out_path.open("w") as f_out:
        for i, line in enumerate(f_in):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            prompt = row["question"]
            try:
                det = detector.detect(prompt)
                if judge_agent is not None:
                    det = judge_agent.maybe_escalate(det)
            except Exception as e:
                logger.error(
                    "Row failed.",
                    extra={LOG_EXTRA_KEY: {"row_id": row.get("id"), "error": str(e)}},
                )
                continue

            f_out.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "question": prompt,
                        "ground_truth": row.get("ground_truth_value")
                        or row.get("expected_behavior"),
                        "ground_truth_aliases": row.get("ground_truth_aliases", []),
                        "greedy_answer": det.greedy_answer,
                        "final_answer": det.final_answer,
                        "final_answer_source": str(det.final_answer_source),
                        "semantic_entropy": det.semantic_entropy,
                        "predictive_entropy": det.predictive_entropy,
                        "escalated": det.escalated,
                        "n_clusters": len(det.clusters),
                        "samples": det.samples,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    run(args.in_path, args.out_path, judge=not args.no_judge, limit=args.limit)
