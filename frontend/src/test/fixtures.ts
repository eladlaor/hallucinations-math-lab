/** Canned API payloads shaped like the backend's responses. */

import type {
  AurocReport,
  ComparisonReport,
  DetectionResult,
  Preset,
  ScoreResponse,
  Status,
} from '@/lib/api'

export const STATUS: Status = {
  provider: 'hf_transformers',
  model_id: 'google/gemma-4-E2B-it',
  judge_configured: true,
  judge_unavailable_reason: null,
  se_threshold: 1.0,
  nli_model_id: 'microsoft/deberta-large-mnli',
}

export const STATUS_NO_JUDGE: Status = {
  ...STATUS,
  judge_configured: false,
  judge_unavailable_reason: 'GCP_PROJECT_ID is not set in .env.',
}

const DIVERGENT_TEXTS = [
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
    question: 'Who was the first woman to walk on Mars?',
    samples: DIVERGENT_TEXTS.map((text) => ({
      text,
      token_logprobs: [{ token: text, logprob: -0.4 }],
    })),
  },
]

export const DIVERGENT_SCORE: ScoreResponse = {
  semantic_entropy: 1.8343719702816235,
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

export const TYPED_SCORE: ScoreResponse = {
  semantic_entropy: 0.6931471805599453,
  predictive_entropy: null,
  clusters: [
    { member_indices: [0], representative_text: 'Paris' },
    { member_indices: [1], representative_text: 'Lyon' },
  ],
  n_answers: 2,
  se_threshold: 1.0,
}

export const ESCALATED_CHAT: DetectionResult = {
  prompt: 'Who was the first woman to walk on Mars?',
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
