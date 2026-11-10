"""The web playground's FastAPI app: a thin layer over the library and the eval scripts.

`create_app` takes every heavy dependency as an argument — the NLI scorer, and
factories for the model under test and the judge — so tests pass fakes and
`python -m demo.web` passes the real ones. The model and the judge are built
lazily, once, on the first request that needs them; DeBERTa loads lazily
inside `NLIScorer` on its first comparison.
"""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from demo.web.api_schemas import (
    AskRequest,
    Preset,
    ResultsListing,
    ResultsReport,
    ScoreRequest,
    ScoreResponse,
    StatusResponse,
)
from demo.web.constants import (
    API_PREFIX,
    INDEX_HTML,
    ROUTE_ASK,
    ROUTE_PRESETS,
    ROUTE_RESULTS,
    ROUTE_RESULTS_REPORT,
    ROUTE_SCORE,
    ROUTE_STATUS,
    SE_TRACE_DETECTORS,
    STATIC_DIR,
    STATIC_MOUNT_NAME,
    STATIC_MOUNT_PATH,
    AskMode,
)
from demo.web.presets import load_presets
from demo.web.results_reports import ResultsPathError, build_report, list_results
from hallucinations_math_lab import (
    DetectionResult,
    DetectorName,
    DirectionalEntailmentScorer,
    DiscreteSemanticEntropyDetector,
    EntailmentScorer,
    Judge,
    NLILabel,
    PredictiveEntropyDetector,
    ResponseEvaluator,
    Sample,
    SampleSet,
    SamplingProvider,
    SemanticEntropyDetector,
    evaluate_responses,
)
from hallucinations_math_lab.config import Settings, get_settings
from hallucinations_math_lab.constants import LOG_EXTRA_KEY
from hallucinations_math_lab.evaluator import _build_default_detectors
from hallucinations_math_lab.services.detector_protocol import Detector

logger = logging.getLogger(__name__)

JUDGE_UNAVAILABLE_REASON = (
    "GCP_PROJECT_ID is not set in .env. The Gemini judge needs it, plus Google "
    "Application Default Credentials, to answer escalated questions."
)
SE_TRACE_NEEDS_DIRECTION_DETAIL = (
    "se_trace needs an NLI scorer that labels one direction at a time "
    "(classify(premise, hypothesis)); the configured scorer has only bidirectional_entails."
)


SE_TRACE_NEEDS_SE_DETAIL = (
    "se_trace needs Semantic Entropy, but ENABLED_DETECTORS {enabled} has neither {se_detectors}."
)


class LockedEntailmentScorer:
    """Serializes calls into one NLI model shared by concurrent requests.

    The first call loads DeBERTa; without the lock two simultaneous first
    requests would each load it.
    """

    def __init__(self, scorer: EntailmentScorer) -> None:
        self._scorer = scorer
        self._lock = threading.Lock()

    def bidirectional_entails(self, a: str, b: str) -> bool:
        with self._lock:
            return self._scorer.bidirectional_entails(a, b)

    @property
    def supports_direction(self) -> bool:
        """Whether the wrapped scorer labels one direction at a time, as the SE trace needs."""
        return isinstance(self._scorer, DirectionalEntailmentScorer)

    def classify(self, premise: str, hypothesis: str) -> NLILabel:
        """One direction, for the SE trace. Raises TypeError if the wrapped scorer has none."""
        scorer = self._scorer
        if not isinstance(scorer, DirectionalEntailmentScorer):
            raise TypeError(
                f"The SE trace needs per-direction NLI labels, but {type(scorer).__name__} "
                "has no classify(premise, hypothesis)."
            )
        with self._lock:
            return scorer.classify(premise, hypothesis)


class LazyOnce[T]:
    """Builds a value on first use, once, under a lock. A failed build is not cached."""

    def __init__(self, factory: Callable[[], T], what: str) -> None:
        self._factory = factory
        self._what = what
        self._value: T | None = None
        self._lock = threading.Lock()

    def get(self) -> T:
        with self._lock:
            if self._value is None:
                try:
                    self._value = self._factory()
                except Exception as e:
                    raise RuntimeError(f"Could not build the {self._what}: {e!s}") from e
                logger.info("Built lazily.", extra={LOG_EXTRA_KEY: {"what": self._what}})
            return self._value


def build_ask_detectors(
    enabled: list[DetectorName], model_id: str, scorer: EntailmentScorer
) -> list[Detector]:
    """The detectors `demo.cli` would run (ENABLED_DETECTORS), with both SE
    estimators on the shared scorer (its lock, and its `classify` for the trace)."""
    detectors: list[Detector] = []
    for name in enabled:
        if name == DetectorName.SEMANTIC_ENTROPY:
            detectors.append(SemanticEntropyDetector(scorer=scorer))
        elif name == DetectorName.DISCRETE_SEMANTIC_ENTROPY:
            detectors.append(DiscreteSemanticEntropyDetector(scorer=scorer))
        else:
            detectors.append(_build_default_detectors([name], model_id)[0])
    return detectors


def judge_unavailable_reason(settings: Settings) -> str | None:
    """Why `chat` cannot escalate to Gemini, or None when it can."""
    return None if settings.gcp_project_id else JUDGE_UNAVAILABLE_REASON


def _api_router(
    *,
    scorer: LockedEntailmentScorer,
    provider: LazyOnce[SamplingProvider],
    judge: LazyOnce[Judge],
    results_dir: Path,
    settings: Settings,
) -> APIRouter:
    router = APIRouter(prefix=API_PREFIX)
    model_lock = threading.Lock()

    def refuse_trace_without_direction(ctx: dict[str, object]) -> None:
        """400 before any work when a trace is asked of a scorer that cannot label one
        direction at a time, instead of a 500 from inside the clustering."""
        if scorer.supports_direction:
            return
        logger.warning(
            "SE trace refused: the NLI scorer has no classify.", extra={LOG_EXTRA_KEY: ctx}
        )
        raise HTTPException(status_code=400, detail=SE_TRACE_NEEDS_DIRECTION_DETAIL)

    evaluators: dict[AskMode, ResponseEvaluator] = {}

    def evaluator_for(mode: AskMode) -> ResponseEvaluator:
        """The mode's evaluator, built on first use and reused, so its build-time
        preflight (probe fit, SEP threshold report) runs once. Call under `model_lock`."""
        if mode not in evaluators:
            slm = provider.get()
            evaluators[mode] = ResponseEvaluator(
                slm=slm,
                detectors=build_ask_detectors(settings.enabled_detectors, slm.model_id, scorer),
                escalation_detector=settings.escalation_detector,
                escalation_threshold=settings.escalation_threshold_for(
                    settings.escalation_detector
                ),
                judge=judge.get() if mode == AskMode.CHAT else None,
            )
        return evaluators[mode]

    @router.get(ROUTE_STATUS, response_model=StatusResponse)
    def status() -> StatusResponse:
        reason = judge_unavailable_reason(settings)
        return StatusResponse(
            provider=settings.slm_provider,
            model_id=settings.vllm_model_id,
            judge_configured=reason is None,
            judge_unavailable_reason=reason,
            se_threshold=settings.se_threshold,
            sep_threshold=settings.sep_threshold,
            escalation_threshold=settings.escalation_threshold_for(settings.escalation_detector),
            nli_model_id=settings.nli_model_id,
        )

    @router.get(ROUTE_PRESETS, response_model=list[Preset])
    def presets() -> list[Preset]:
        try:
            return load_presets()
        except RuntimeError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.post(ROUTE_SCORE, response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        has_logprobs = all(a.token_logprobs for a in request.answers)
        ctx = {
            "n_answers": len(request.answers),
            "has_logprobs": has_logprobs,
            "se_trace": request.se_trace,
            "question": request.question[:80],
        }
        if request.se_trace:
            refuse_trace_without_direction(ctx)
        logger.info("Scoring answers.", extra={LOG_EXTRA_KEY: ctx})
        started = time.perf_counter()
        # Likelihood-weighted SE needs every answer's logprobs; typed answers
        # get discrete SE, and the response names which one ran.
        se = (
            SemanticEntropyDetector(scorer=scorer)
            if has_logprobs
            else DiscreteSemanticEntropyDetector(scorer=scorer)
        )
        detectors: list[Detector] = [se]
        if has_logprobs:
            detectors.append(PredictiveEntropyDetector())
        samples = [
            Sample(text=a.text, token_logprobs=a.token_logprobs or []) for a in request.answers
        ]
        try:
            result = evaluate_responses(
                prompt=request.question,
                samples=SampleSet(prompt=request.question, samples=samples),
                detectors=detectors,
                escalation_detector=se.name,
                escalation_threshold=settings.escalation_threshold_for(se.name),
                greedy_answer=samples[0].text,
                trace_semantic_entropy=request.se_trace,
            )
        except Exception as e:
            logger.error("Scoring failed.", extra={LOG_EXTRA_KEY: {**ctx, "error": str(e)}})
            raise HTTPException(status_code=500, detail=str(e)) from e
        if result.clusters is None:  # unreachable: SE always runs here
            raise HTTPException(status_code=500, detail="SE returned no clusters.")
        logger.info(
            "Scored answers.",
            extra={
                LOG_EXTRA_KEY: {
                    **ctx,
                    "n_clusters": len(result.clusters),
                    "elapsed_s": round(time.perf_counter() - started, 3),
                }
            },
        )
        return ScoreResponse(
            semantic_entropy=result.detector_scores[se.name],
            se_estimator=se.estimator,
            predictive_entropy=result.detector_scores.get(DetectorName.PREDICTIVE_ENTROPY),
            clusters=result.clusters,
            n_answers=len(samples),
            se_threshold=settings.se_threshold,
            se_trace=result.se_trace,
        )

    @router.post(ROUTE_ASK, response_model=DetectionResult)
    def ask(request: AskRequest) -> DetectionResult:
        ctx = {
            "mode": str(request.mode),
            "se_trace": request.se_trace,
            "question": request.question[:80],
        }
        if request.se_trace and not SE_TRACE_DETECTORS & set(settings.enabled_detectors):
            logger.warning("SE trace refused: SE is not enabled.", extra={LOG_EXTRA_KEY: ctx})
            raise HTTPException(
                status_code=400,
                detail=SE_TRACE_NEEDS_SE_DETAIL.format(
                    enabled=[str(d) for d in settings.enabled_detectors],
                    se_detectors=" nor ".join(sorted(SE_TRACE_DETECTORS)),
                ),
            )
        if request.se_trace:
            refuse_trace_without_direction(ctx)
        if request.mode == AskMode.CHAT:
            reason = judge_unavailable_reason(settings)
            if reason is not None:
                logger.warning("Chat refused: no judge.", extra={LOG_EXTRA_KEY: ctx})
                raise HTTPException(status_code=400, detail=reason)
        logger.info("Asking the model.", extra={LOG_EXTRA_KEY: ctx})
        started = time.perf_counter()
        try:
            with model_lock:
                result = evaluator_for(request.mode).evaluate(
                    request.question, trace_semantic_entropy=request.se_trace
                )
        except Exception as e:
            logger.error("Ask failed.", extra={LOG_EXTRA_KEY: {**ctx, "error": str(e)}})
            raise HTTPException(status_code=500, detail=str(e)) from e
        logger.info(
            "Asked the model.",
            extra={
                LOG_EXTRA_KEY: {
                    **ctx,
                    "escalated": result.escalated,
                    "source": str(result.final_answer_source),
                    "elapsed_s": round(time.perf_counter() - started, 3),
                }
            },
        )
        return result

    @router.get(ROUTE_RESULTS, response_model=ResultsListing)
    def results() -> ResultsListing:
        try:
            return list_results(results_dir)
        except OSError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get(ROUTE_RESULTS_REPORT, response_model=ResultsReport)
    def results_report(path: str = Query(min_length=1)) -> ResultsReport:
        try:
            return build_report(results_dir, path)
        except ResultsPathError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return router


def create_app(
    *,
    nli_scorer: EntailmentScorer,
    provider_factory: Callable[[], SamplingProvider],
    judge_factory: Callable[[], Judge],
    results_dir: Path,
    static_dir: Path = STATIC_DIR,
    settings: Settings | None = None,
) -> FastAPI:
    """The playground app: the JSON API under /api, the built UI at /.

    Raises FileNotFoundError when `static_dir` holds no built UI.
    """
    if not (static_dir / INDEX_HTML).is_file():
        raise FileNotFoundError(
            f"No built UI at {static_dir / INDEX_HTML}. Build it with `npm run build` in "
            "frontend/ (see the Developer Guide)."
        )
    resolved_settings = settings if settings is not None else get_settings()
    resolved_results = results_dir.resolve()

    app = FastAPI(title="hallucinations-math-lab playground")
    app.include_router(
        _api_router(
            scorer=LockedEntailmentScorer(nli_scorer),
            provider=LazyOnce(provider_factory, "model under test"),
            judge=LazyOnce(judge_factory, "judge"),
            results_dir=resolved_results,
            settings=resolved_settings,
        )
    )
    app.mount(
        STATIC_MOUNT_PATH, StaticFiles(directory=static_dir, html=True), name=STATIC_MOUNT_NAME
    )
    logger.info(
        "Playground app created.",
        extra={
            LOG_EXTRA_KEY: {
                "static_dir": str(static_dir),
                "results_dir": str(resolved_results),
                "provider": resolved_settings.slm_provider,
                "model_id": resolved_settings.vllm_model_id,
            }
        },
    )
    return app
