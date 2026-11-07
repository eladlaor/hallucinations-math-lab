/** Pure helpers for displaying scores and deciding escalation in the browser. */

import {
  DetectorName,
  PERCENT_DECIMALS,
  SCORE_DECIMALS,
  ScoreScale,
  SeEstimator,
  THRESHOLD_MIN_MAX_NATS,
} from '@/lib/constants'

/** Three decimals, as the example script prints them; never "-0.000". */
export function formatScore(value: number, decimals: number = SCORE_DECIMALS): string {
  const factor = 10 ** decimals
  return (Math.round(value * factor) / factor + 0).toFixed(decimals)
}

/** Escalation is just SE > threshold, so the slider never needs the server. */
export function shouldEscalate(semanticEntropy: number, threshold: number): boolean {
  return semanticEntropy > threshold
}

/** SE peaks at ln(N) when every answer means something different. */
export function thresholdSliderMax(nAnswers: number): number {
  const maxEntropy = nAnswers > 1 ? Math.log(nAnswers) : 0
  return Math.max(THRESHOLD_MIN_MAX_NATS, Math.ceil(maxEntropy * 10) / 10)
}

export interface DetectorInfo {
  short: string
  name: string
  meaning: string
  scale: ScoreScale
}

const DETECTOR_INFO: Record<string, DetectorInfo> = {
  [DetectorName.SEMANTIC_ENTROPY]: {
    short: 'SE',
    name: 'Semantic Entropy',
    meaning:
      'How many different meanings the answers spread across, in nats. 0 means they all say the same thing.',
    scale: ScoreScale.NUMBER,
  },
  [DetectorName.DISCRETE_SEMANTIC_ENTROPY]: {
    short: 'SE, discrete',
    name: 'Discrete Semantic Entropy',
    meaning:
      'Semantic Entropy with each meaning weighed by how many answers say it, for answers without token probabilities.',
    scale: ScoreScale.NUMBER,
  },
  [DetectorName.PREDICTIVE_ENTROPY]: {
    short: 'PE',
    name: 'Predictive Entropy',
    meaning:
      "How unsure the model was of its own words, from token probabilities. Varied wording reads as uncertainty, even when the meaning doesn't change.",
    scale: ScoreScale.NUMBER,
  },
  [DetectorName.SEMANTIC_ENTROPY_PROBE]: {
    short: 'SEP',
    name: 'Semantic Entropy Probe',
    meaning:
      "P(high SE): a linear probe's estimate, from one hidden state and without repeated sampling, of the chance that the sampled answers would spread across many meanings. It is not the chance that the answer is wrong.",
    scale: ScoreScale.PROBABILITY,
  },
}

const SE_ESTIMATOR_NOTE: Record<SeEstimator, string> = {
  [SeEstimator.LIKELIHOOD_WEIGHTED]:
    'Each meaning is weighed by how likely the model found the answers that say it.',
  [SeEstimator.DISCRETE]:
    'Each meaning is weighed by how many answers say it: typed or edited answers carry no token probabilities to weigh by.',
}

/** One sentence on how SE weighed the meanings, for the score's note. */
export function seEstimatorNote(estimator: SeEstimator): string {
  return SE_ESTIMATOR_NOTE[estimator]
}

export function detectorInfo(name: string): DetectorInfo {
  return DETECTOR_INFO[name] ?? { short: name, name, meaning: '', scale: ScoreScale.NUMBER }
}

/**
 * A score or threshold on its detector's scale: a percentage with
 * PERCENT_DECIMALS places for a probability (SEP), otherwise a plain number
 * with `numberDecimals` places. `numberDecimals` never applies to a probability.
 */
export function formatDetectorScore(
  name: string,
  value: number,
  numberDecimals: number = SCORE_DECIMALS,
): string {
  if (detectorInfo(name).scale === ScoreScale.PROBABILITY) {
    return `${formatScore(value * 100, PERCENT_DECIMALS)}%`
  }
  return formatScore(value, numberDecimals)
}

/** Canonical display order: PE, SEP, SE, then anything else alphabetically. */
const DETECTOR_ORDER: string[] = [
  DetectorName.PREDICTIVE_ENTROPY,
  DetectorName.SEMANTIC_ENTROPY_PROBE,
  DetectorName.SEMANTIC_ENTROPY,
]

export function sortDetectors(names: string[]): string[] {
  const rank = (n: string) => {
    const i = DETECTOR_ORDER.indexOf(n)
    return i === -1 ? DETECTOR_ORDER.length : i
  }
  return [...names].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
}

/** "1:05" from milliseconds. */
export function formatElapsed(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}
