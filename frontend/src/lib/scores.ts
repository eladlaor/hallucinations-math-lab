/** Pure helpers for displaying scores and deciding escalation in the browser. */

import { DetectorName, SCORE_DECIMALS, THRESHOLD_MIN_MAX_NATS } from '@/lib/constants'

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
}

const DETECTOR_INFO: Record<string, DetectorInfo> = {
  [DetectorName.SEMANTIC_ENTROPY]: {
    short: 'SE',
    name: 'Semantic Entropy',
    meaning:
      'How many different meanings the answers spread across, in nats. 0 means they all say the same thing.',
  },
  [DetectorName.PREDICTIVE_ENTROPY]: {
    short: 'PE',
    name: 'Predictive Entropy',
    meaning:
      "How unsure the model was of its own words, from token probabilities. Varied wording reads as uncertainty, even when the meaning doesn't change.",
  },
  [DetectorName.SEMANTIC_ENTROPY_PROBE]: {
    short: 'SEP',
    name: 'Semantic Entropy Probe',
    meaning: "A linear probe's prediction of SE from one hidden state, without repeated sampling.",
  },
}

export function detectorInfo(name: string): DetectorInfo {
  return DETECTOR_INFO[name] ?? { short: name, name, meaning: '' }
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
