"""`uv run python -m demo.web` — serve the playground at http://127.0.0.1:8765.

Binds 127.0.0.1 only. The real dependencies are wired here: DeBERTa through
the library's shared NLIScorer, the model under test from SLM_PROVIDER /
VLLM_MODEL_ID, and the Gemini judge. All three load on first use, not at
startup, so the server starts in seconds.
"""

import argparse
import logging

import uvicorn

from demo.web.app import create_app
from demo.web.constants import DEFAULT_HOST, DEFAULT_PORT, RESULTS_DIR
from hallucinations_math_lab.constants import LOG_EXTRA_KEY
from hallucinations_math_lab.services.nli import get_nli_scorer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local web playground.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port on 127.0.0.1.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # Lazy imports: the demo provider and the judge pull in optional extras.
    from demo.slm_provider import build_default_provider
    from hallucinations_math_lab.judges import GeminiJudge

    try:
        app = create_app(
            nli_scorer=get_nli_scorer(),
            provider_factory=build_default_provider,
            judge_factory=GeminiJudge,
            results_dir=RESULTS_DIR,
        )
    except Exception as e:
        logger.error("Could not start the playground.", extra={LOG_EXTRA_KEY: {"error": str(e)}})
        raise
    print(f"Web playground: http://{DEFAULT_HOST}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=DEFAULT_HOST, port=args.port)


if __name__ == "__main__":
    main()
