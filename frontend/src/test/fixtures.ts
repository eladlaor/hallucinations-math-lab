/** Canned API payloads shaped like the backend's responses. */

import type {
  AurocReport,
  ClusteringComparison,
  ComparisonReport,
  DetectionResult,
  Preset,
  ScoreResponse,
  SemanticEntropyTrace,
  Status,
} from '@/lib/api'
import { NliLabel, SeEstimator } from '@/lib/constants'

export const STATUS: Status = {
  provider: 'hf_transformers',
  model_id: 'google/gemma-4-E2B-it',
  judge_configured: true,
  judge_unavailable_reason: null,
  se_threshold: 1.0,
  sep_threshold: 0.5,
  escalation_threshold: 1.0,
  nli_model_id: 'microsoft/deberta-large-mnli',
  se_trace_unavailable_reason: null,
  ask_se_trace_unavailable_reason: null,
}

const NO_CLASSIFY_REASON =
  'se_trace needs an NLI scorer that labels one direction at a time (classify(premise, hypothesis)); the configured scorer has only bidirectional_entails.'

/** A server whose NLI scorer has no per-direction classify, so it cannot record the trace. */
export const STATUS_NO_TRACE: Status = {
  ...STATUS,
  se_trace_unavailable_reason: NO_CLASSIFY_REASON,
  ask_se_trace_unavailable_reason: NO_CLASSIFY_REASON,
}

/** A server whose ENABLED_DETECTORS has no SE detector: Score can trace, Ask cannot. */
export const STATUS_NO_SE: Status = {
  ...STATUS,
  ask_se_trace_unavailable_reason:
    "se_trace needs Semantic Entropy, but ENABLED_DETECTORS ['predictive_entropy'] has neither discrete_semantic_entropy nor semantic_entropy.",
}

export const STATUS_NO_JUDGE: Status = {
  ...STATUS,
  judge_configured: false,
  judge_unavailable_reason: 'GCP_PROJECT_ID is not set in .env.',
}

export const DIVERGENT_TEXTS = [
  'Sally Ride',
  'Valentina Tereshkova',
  'Peggy Whitson',
  'Sally Ride',
  'Christina Koch',
  'Mae Jemison',
  'No woman has walked on Mars.',
  'Valentina Tereshkova',
  'Eileen Collins',
  'Sally Ride',
]

export const DIVERGENT_QUESTION = 'Who was the first woman to walk on Mars?'

export const PRESETS: Preset[] = [
  {
    name: 'paraphrased',
    question: 'What is the capital of France?',
    samples: ['Paris', 'The capital is Paris.'].map((text) => ({
      text,
      token_logprobs: [{ token: text, logprob: -0.6 }],
    })),
  },
  {
    name: 'divergent',
    question: DIVERGENT_QUESTION,
    samples: DIVERGENT_TEXTS.map((text) => ({
      text,
      token_logprobs: [{ token: text, logprob: -0.4 }],
    })),
  },
]

type Verdicts = [NliLabel, NliLabel | null]
const NO_MATCH: Verdicts = [NliLabel.CONTRADICTION, null]

/**
 * The comparisons the backend's greedy clustering runs to reach `members`:
 * each answer against each existing cluster's first answer, in order, stopping
 * at the first join. A failed comparison is contradiction unless `verdicts`
 * (keyed "answer-cluster") says otherwise.
 */
function clusteringTrace(
  members: number[][],
  verdicts: Record<string, Verdicts> = {},
): SemanticEntropyTrace['clustering'] {
  const nAnswers = members.flat().length
  const clusterOf = (answer: number) => members.findIndex((m) => m.includes(answer))
  const comparisons: ClusteringComparison[] = []
  const placements = []
  let nOpen = 0
  for (let answer = 0; answer < nAnswers; answer += 1) {
    const target = clusterOf(answer)
    const opened = target === nOpen
    const compared = opened ? nOpen : target + 1
    for (let cluster = 0; cluster < compared; cluster += 1) {
      const joined = cluster === target
      const [forward, backward] = joined
        ? [NliLabel.ENTAILMENT, NliLabel.ENTAILMENT]
        : (verdicts[`${answer}-${cluster}`] ?? NO_MATCH)
      comparisons.push({
        answer_index: answer,
        cluster_index: cluster,
        representative_index: members[cluster]?.[0] ?? -1,
        forward,
        backward,
        joined,
      })
    }
    if (opened) nOpen += 1
    placements.push({ answer_index: answer, cluster_index: target, opened_new_cluster: opened })
  }
  return { comparisons, placements }
}

const DIVERGENT_MEMBERS = [[0, 3, 9], [1, 7], [2], [4], [5], [6], [8]]

/**
 * The divergent preset's clustering, hand-built. In the real run (captured in
 * capturedScoreTraces.ts) every failed comparison is contradiction; two are
 * edited here to exercise the replay's neutral and one-way branches, which
 * replay.test.ts checks are the only differences. All ten answers carry the
 * same logprob, so p(c) is |c| / N here too.
 */
export const DIVERGENT_TRACE: SemanticEntropyTrace = {
  clustering: clusteringTrace(DIVERGENT_MEMBERS, {
    // Edited: "No woman has walked on Mars." vs "Sally Ride" neutral, so the reverse never runs.
    '6-0': [NliLabel.NEUTRAL, null],
    // Edited: "Eileen Collins" vs "Sally Ride" one way only, so no join.
    '8-0': [NliLabel.ENTAILMENT, NliLabel.NEUTRAL],
  }),
  estimator: SeEstimator.LIKELIHOOD_WEIGHTED,
  answer_likelihoods: Array.from({ length: 10 }, () => Math.exp(-0.4)),
  cluster_probabilities: [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
}

export const DIVERGENT_SCORE: ScoreResponse = {
  semantic_entropy: 1.8343719702816235,
  se_estimator: SeEstimator.LIKELIHOOD_WEIGHTED,
  predictive_entropy: 0.4,
  clusters: [
    { member_indices: [0, 3, 9], representative_text: 'Sally Ride' },
    { member_indices: [1, 7], representative_text: 'Valentina Tereshkova' },
    { member_indices: [2], representative_text: 'Peggy Whitson' },
    { member_indices: [4], representative_text: 'Christina Koch' },
    { member_indices: [5], representative_text: 'Mae Jemison' },
    { member_indices: [6], representative_text: 'No woman has walked on Mars.' },
    { member_indices: [8], representative_text: 'Eileen Collins' },
  ],
  n_answers: 10,
  se_threshold: 1.0,
}

export const DIVERGENT_SCORE_TRACED: ScoreResponse = { ...DIVERGENT_SCORE, se_trace: DIVERGENT_TRACE }

export const TYPED_SCORE: ScoreResponse = {
  semantic_entropy: 0.6931471805599453,
  se_estimator: SeEstimator.DISCRETE,
  predictive_entropy: null,
  clusters: [
    { member_indices: [0], representative_text: 'Paris' },
    { member_indices: [1], representative_text: 'Lyon' },
  ],
  n_answers: 2,
  se_threshold: 1.0,
}

export const TYPED_TRACE: SemanticEntropyTrace = {
  clustering: clusteringTrace([[0], [1]]),
  estimator: SeEstimator.DISCRETE,
  answer_likelihoods: null,
  cluster_probabilities: [0.5, 0.5],
}

export const TYPED_SCORE_TRACED: ScoreResponse = { ...TYPED_SCORE, se_trace: TYPED_TRACE }

export const WEIGHTED_QUESTION = 'What is the capital of Australia?'
export const WEIGHTED_TEXTS = ['Canberra', 'Canberra', 'Sydney']

/**
 * Likelihood-weighted SE whose weights are not the cluster sizes: the lone
 * "Sydney" carries 0.6 of the likelihood, so its p(c) is 0.6 where |c| / N
 * would give 1/3. Catches a weight summed over the wrong answers.
 */
export const WEIGHTED_TRACE: SemanticEntropyTrace = {
  clustering: clusteringTrace([[0, 1], [2]]),
  estimator: SeEstimator.LIKELIHOOD_WEIGHTED,
  answer_likelihoods: [0.2, 0.2, 0.6],
  cluster_probabilities: [0.4, 0.6],
}

export const WEIGHTED_SCORE_TRACED: ScoreResponse = {
  semantic_entropy: 0.6730116670092565,
  se_estimator: SeEstimator.LIKELIHOOD_WEIGHTED,
  predictive_entropy: 1.2432338162113972, // mean of -ln(likelihood)
  clusters: [
    { member_indices: [0, 1], representative_text: 'Canberra' },
    { member_indices: [2], representative_text: 'Sydney' },
  ],
  n_answers: 3,
  se_threshold: 1.0,
  se_trace: WEIGHTED_TRACE,
}

export const ESCALATED_CHAT: DetectionResult = {
  prompt: DIVERGENT_QUESTION,
  greedy_answer: 'Sally Ride',
  samples: DIVERGENT_TEXTS,
  detector_scores: { predictive_entropy: 0.4, semantic_entropy: 1.834 },
  clusters: DIVERGENT_SCORE.clusters,
  escalation_detector: 'semantic_entropy',
  escalation_threshold: 1.0,
  escalated: true,
  final_answer: 'No woman has walked on Mars.',
  final_answer_source: 'judge_model',
  judge_reasoning: 'No human has landed on Mars.',
  escalation_policy_desc: null,
}

export const ESCALATED_CHAT_TRACED: DetectionResult = {
  ...ESCALATED_CHAT,
  detector_scores: { ...ESCALATED_CHAT.detector_scores, semantic_entropy: DIVERGENT_SCORE.semantic_entropy },
  se_trace: DIVERGENT_TRACE,
}

/** A Detect run escalated by SEP: its threshold is the status's sep_threshold. */
export const SEP_DETECT: DetectionResult = {
  ...ESCALATED_CHAT,
  detector_scores: { semantic_entropy_probe: 0.7234, semantic_entropy: 1.834 },
  escalation_detector: 'semantic_entropy_probe',
  escalation_threshold: STATUS.sep_threshold,
  final_answer: 'Sally Ride',
  final_answer_source: 'small_model_greedy',
  judge_reasoning: null,
}

const SE_AUROC = {
  auroc: 0.8123,
  ci: { level: 0.95, low: 0.751, high: 0.8702, n_resamples: 1000, n_degenerate_resamples: 0 },
}

export const RUN_REPORT: AurocReport = {
  run_config: { model_id: 'google/gemma-4-E2B-it', sample_n: 10, sample_temperature: 1.0 },
  n_rows: 200,
  n_hallucinations: 87,
  detectors: {
    semantic_entropy: SE_AUROC,
    predictive_entropy: {
      auroc: 0.7011,
      ci: { level: 0.95, low: 0.6302, high: 0.7688, n_resamples: 1000, n_degenerate_resamples: 0 },
    },
  },
  criteria: {
    se_auroc: {
      value: 0.8123,
      ci: { level: 0.95, low: 0.751, high: 0.8702, n_resamples: 1000, n_degenerate_resamples: 0 },
      target: 0.75,
      met: true,
    },
  },
}

export const SWEEP_REPORT: ComparisonReport = {
  n_questions: 200,
  sampling: { sample_n: 10, sample_temperature: 1.0 },
  models: [
    {
      model_id: 'google/gemma-4-E2B-it',
      source: 'sweep/a.jsonl',
      n_rows: 200,
      n_hallucinations: 87,
      accuracy: 0.565,
      detectors: RUN_REPORT.detectors,
    },
    {
      model_id: 'google/gemma-4-E4B-it',
      source: 'sweep/b.jsonl',
      n_rows: 200,
      n_hallucinations: 61,
      accuracy: 0.695,
      detectors: { semantic_entropy: SE_AUROC },
    },
  ],
}
