/**
 * Typed client for the playground API (demo/web/app.py).
 *
 * Every call logs start / success / failure with its duration. A non-2xx
 * response throws ApiError carrying the backend's message verbatim, so the UI
 * can show exactly what went wrong — no silent retry.
 */

import {
  ApiRoute,
  ASK_TIMEOUT_MS,
  DEFAULT_TIMEOUT_MS,
  ResultsKind,
  type AnswerSource,
  type AskMode,
  type SeEstimator,
} from '@/lib/constants'
import { logger } from '@/lib/logger'

export interface TokenLogprob {
  token: string
  logprob: number
}

export interface Sample {
  text: string
  token_logprobs: TokenLogprob[]
}

export interface Cluster {
  member_indices: number[]
  representative_text: string
}

export interface Status {
  provider: string
  model_id: string
  judge_configured: boolean
  judge_unavailable_reason: string | null
  se_threshold: number
  nli_model_id: string
}

export interface Preset {
  name: string
  question: string
  samples: Sample[]
}

export interface ScoreAnswer {
  text: string
  token_logprobs?: TokenLogprob[] | null
}

export interface ScoreRequest {
  question: string
  answers: ScoreAnswer[]
}

export interface ScoreResponse {
  semantic_entropy: number
  se_estimator: SeEstimator
  predictive_entropy: number | null
  clusters: Cluster[]
  n_answers: number
  se_threshold: number
}

export interface DetectionResult {
  prompt: string
  greedy_answer: string
  samples: string[]
  detector_scores: Record<string, number>
  clusters: Cluster[] | null
  escalation_detector: string
  escalation_threshold: number
  escalated: boolean
  final_answer: string
  final_answer_source: AnswerSource
  judge_reasoning: string | null
  escalation_policy_desc: string | null
}

export interface ResultsEntry {
  path: string
  kind: ResultsKind
  n_files: number
}

export interface ResultsListing {
  results_dir: string
  entries: ResultsEntry[]
}

export interface ConfidenceInterval {
  level: number
  low: number
  high: number
  n_resamples: number
  n_degenerate_resamples: number
}

export interface DetectorAuroc {
  auroc: number
  ci: ConfidenceInterval
}

export interface Criterion {
  value: number
  ci: ConfidenceInterval
  target: number
  met: boolean
}

export interface AurocReport {
  run_config: { model_id: string; sample_n: number; sample_temperature: number }
  n_rows: number
  n_hallucinations: number
  detectors: Record<string, DetectorAuroc>
  criteria?: Record<string, Criterion>
}

export interface ModelSummary {
  model_id: string
  source: string
  n_rows: number
  n_hallucinations: number
  accuracy: number
  detectors: Record<string, DetectorAuroc>
}

export interface ComparisonReport {
  n_questions: number
  sampling: { sample_n: number; sample_temperature: number }
  models: ModelSummary[]
}

export type ResultsReport =
  | { path: string; kind: typeof ResultsKind.RUN; report: AurocReport }
  | { path: string; kind: typeof ResultsKind.SWEEP; report: ComparisonReport }

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

interface ValidationIssue {
  loc?: (string | number)[]
  msg?: string
}

/** FastAPI puts the message in `detail`: a string, or a list of validation issues. */
async function readErrorMessage(response: Response): Promise<string> {
  const text = await response.text()
  try {
    const body = JSON.parse(text) as { detail?: string | ValidationIssue[] }
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail)) {
      return body.detail
        .map((issue) => `${(issue.loc ?? []).join('.')}: ${issue.msg ?? 'invalid'}`)
        .join('; ')
    }
  } catch {
    // Not JSON: fall through to the raw body, which is the message.
  }
  return text || `${response.status} ${response.statusText}`
}

async function request<T>(
  call: string,
  url: string,
  init: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const started = performance.now()
  const elapsedMs = () => Math.round(performance.now() - started)
  logger.info('api call started', { call, url, method: init.method ?? 'GET' })
  try {
    const response = await fetch(url, { ...init, signal: AbortSignal.timeout(timeoutMs) })
    if (!response.ok) {
      throw new ApiError(response.status, await readErrorMessage(response))
    }
    const body = (await response.json()) as T
    logger.info('api call succeeded', { call, url, status: response.status, elapsedMs: elapsedMs() })
    return body
  } catch (error) {
    logger.error('api call failed', {
      call,
      url,
      status: error instanceof ApiError ? error.status : null,
      elapsedMs: elapsedMs(),
      error,
    })
    throw error
  }
}

function postJson(body: unknown): RequestInit {
  return {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }
}

export const api = {
  status: () => request<Status>('status', ApiRoute.STATUS),
  presets: () => request<Preset[]>('presets', ApiRoute.PRESETS),
  score: (body: ScoreRequest) => request<ScoreResponse>('score', ApiRoute.SCORE, postJson(body)),
  ask: (question: string, mode: AskMode) =>
    request<DetectionResult>('ask', ApiRoute.ASK, postJson({ question, mode }), ASK_TIMEOUT_MS),
  results: () => request<ResultsListing>('results', ApiRoute.RESULTS),
  resultsReport: (path: string) =>
    request<ResultsReport>(
      'resultsReport',
      `${ApiRoute.RESULTS_REPORT}?${new URLSearchParams({ path }).toString()}`,
    ),
}
