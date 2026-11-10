/** Immutable values shared by the UI: API routes, enum-like sets, limits, links. */

export const ApiRoute = {
  STATUS: '/api/status',
  PRESETS: '/api/presets',
  SCORE: '/api/score',
  ASK: '/api/ask',
  RESULTS: '/api/results',
  RESULTS_REPORT: '/api/results/report',
} as const

export const AskMode = { DETECT: 'detect', CHAT: 'chat' } as const
export type AskMode = (typeof AskMode)[keyof typeof AskMode]

export const ResultsKind = { RUN: 'run', SWEEP: 'sweep' } as const
export type ResultsKind = (typeof ResultsKind)[keyof typeof ResultsKind]

export const AnswerSource = {
  SMALL_MODEL_GREEDY: 'small_model_greedy',
  JUDGE_MODEL: 'judge_model',
} as const
export type AnswerSource = (typeof AnswerSource)[keyof typeof AnswerSource]

export const DetectorName = {
  PREDICTIVE_ENTROPY: 'predictive_entropy',
  SEMANTIC_ENTROPY_PROBE: 'semantic_entropy_probe',
  SEMANTIC_ENTROPY: 'semantic_entropy',
  DISCRETE_SEMANTIC_ENTROPY: 'discrete_semantic_entropy',
} as const

/** How a detector's score reads: an entropy as a plain number, or a probability as a percentage. */
export const ScoreScale = { NUMBER: 'number', PROBABILITY: 'probability' } as const
export type ScoreScale = (typeof ScoreScale)[keyof typeof ScoreScale]

/** How SE weighed its meaning clusters (the API's `se_estimator`). */
export const SeEstimator = {
  LIKELIHOOD_WEIGHTED: 'likelihood_weighted',
  DISCRETE: 'discrete',
} as const
export type SeEstimator = (typeof SeEstimator)[keyof typeof SeEstimator]

/** One direction's NLI verdict (the API's NLILabel). */
export const NliLabel = {
  ENTAILMENT: 'entailment',
  NEUTRAL: 'neutral',
  CONTRADICTION: 'contradiction',
} as const
export type NliLabel = (typeof NliLabel)[keyof typeof NliLabel]

/** The detector whose score each SE estimator produces, for the replay's last step. */
export const SE_DETECTOR_BY_ESTIMATOR: Record<SeEstimator, string> = {
  [SeEstimator.LIKELIHOOD_WEIGHTED]: DetectorName.SEMANTIC_ENTROPY,
  [SeEstimator.DISCRETE]: DetectorName.DISCRETE_SEMANTIC_ENTROPY,
}

/**
 * The replay's playback speeds, slowest first, as milliseconds per step. The
 * trace carries no timings: "Real time" approximates one DeBERTa comparison on
 * a CPU, it is not a measurement of this run.
 */
export const ReplaySpeed = {
  SLOW_MOTION: 'slow_motion',
  SLOW: 'slow',
  NORMAL: 'normal',
  REAL_TIME: 'real_time',
} as const
export type ReplaySpeed = (typeof ReplaySpeed)[keyof typeof ReplaySpeed]
export const REPLAY_STEP_MS: Record<ReplaySpeed, number> = {
  [ReplaySpeed.SLOW_MOTION]: 4000,
  [ReplaySpeed.SLOW]: 2000,
  [ReplaySpeed.NORMAL]: 1000,
  [ReplaySpeed.REAL_TIME]: 250,
}
export const REPLAY_SPEED_LABEL: Record<ReplaySpeed, string> = {
  [ReplaySpeed.SLOW_MOTION]: 'Slow motion (4 s a step)',
  [ReplaySpeed.SLOW]: 'Slow (2 s a step)',
  [ReplaySpeed.NORMAL]: 'Normal (1 s a step)',
  [ReplaySpeed.REAL_TIME]: 'Real time (about 0.25 s a step)',
}
export const DEFAULT_REPLAY_SPEED: ReplaySpeed = ReplaySpeed.SLOW
/**
 * How far the SE in the trace may drift from the reported SE, and each p(c) from
 * the one its estimator implies, before the replay refuses the trace.
 */
export const REPLAY_SE_TOLERANCE = 1e-6

export const TabId = { SCORE: 'score', ASK: 'ask', RESULTS: 'results' } as const
export type TabId = (typeof TabId)[keyof typeof TabId]

export const RequestState = {
  IDLE: 'idle',
  LOADING: 'loading',
  SUCCESS: 'success',
  ERROR: 'error',
} as const
export type RequestState = (typeof RequestState)[keyof typeof RequestState]

/** Plain requests; a CPU generation on the Ask tab can take many minutes. */
export const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000
export const ASK_TIMEOUT_MS = 60 * 60 * 1000

export const SCORE_DECIMALS = 3
/** A probability score (SEP's P(high SE)) shows as a percentage with one decimal. */
export const PERCENT_DECIMALS = 1
/** The threshold slider covers at least 0..3 nats, more when ln(N) is larger. */
export const THRESHOLD_MIN_MAX_NATS = 3
export const THRESHOLD_STEP_NATS = 0.05
export const ELAPSED_TICK_MS = 1000

export const README_HARDWARE_URL =
  'https://github.com/eladlaor/hallucinations-math-lab#what-hardware-do-i-need-and-why'
export const README_ASK_URL =
  'https://github.com/eladlaor/hallucinations-math-lab#how-do-i-ask-one-question'
export const BENCHMARK_COMMAND =
  'uv run python -m eval.run_benchmark --out eval/results/triviaqa_200.jsonl --no-judge'
