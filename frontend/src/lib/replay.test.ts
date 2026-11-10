import { describe, expect, it } from 'vitest'

import type { ClusteringComparison } from '@/lib/api'
import { NliLabel } from '@/lib/constants'
import { buildReplay, CompareOutcome, nliInput, ReplayStepKind, type ReplayStep } from '@/lib/replay'
import { DIVERGENT_PRESET, PARAPHRASED_PRESET, type CapturedScore } from '@/test/capturedScoreTraces'
import {
  DIVERGENT_SCORE,
  DIVERGENT_TRACE,
  TYPED_SCORE_TRACED,
  TYPED_TRACE,
  WEIGHTED_SCORE_TRACED,
  WEIGHTED_TRACE,
} from '@/test/fixtures'

function divergentReplay() {
  return buildReplay(DIVERGENT_TRACE, DIVERGENT_SCORE.clusters, DIVERGENT_SCORE.semantic_entropy)
}

function membersOf(step: ReplayStep | undefined): number[][] {
  if (!step) throw new Error('no such step')
  return step.clustersSoFar
}

function outcomesFor(answerIndex: number) {
  return divergentReplay().steps.filter(
    (s) => s.kind === ReplayStepKind.COMPARE && s.comparison.answer_index === answerIndex,
  )
}

describe('buildReplay', () => {
  it('plays the divergent trace back to exactly the returned seven meaning clusters', () => {
    const replay = divergentReplay()

    // Answer 1 opens the first cluster, 25 comparisons, then one entropy term per cluster.
    expect(replay.steps).toHaveLength(1 + 25 + 7)
    const lastClustering = replay.steps[1 + 25 - 1]
    expect(membersOf(lastClustering)).toEqual(DIVERGENT_SCORE.clusters.map((c) => c.member_indices))
    expect(membersOf(replay.steps.at(-1))).toEqual(membersOf(lastClustering))
  })

  it('opens the first meaning cluster with nothing to compare against', () => {
    const [first] = divergentReplay().steps

    expect(first).toMatchObject({ kind: ReplayStepKind.FIRST, answerIndex: 0 })
    expect(membersOf(first)).toEqual([[0]])
  })

  it('tries each meaning cluster in order and joins the first that matches both ways', () => {
    const steps = outcomesFor(7)

    expect(steps.map((s) => s.kind === ReplayStepKind.COMPARE && s.outcome)).toEqual([
      CompareOutcome.NEXT,
      CompareOutcome.JOINED,
    ])
    expect(membersOf(steps[1])[1]).toEqual([1, 7])
  })

  it('opens a new meaning cluster after the last failed comparison', () => {
    const steps = outcomesFor(2)

    expect(steps.map((s) => s.kind === ReplayStepKind.COMPARE && s.outcome)).toEqual([
      CompareOutcome.NEXT,
      CompareOutcome.OPENED,
    ])
    expect(membersOf(steps[1])).toEqual([[0], [1], [2]])
  })

  it('fills in -p ln p one meaning cluster at a time, ending at the reported SE', () => {
    const entropy = divergentReplay().steps.filter((s) => s.kind === ReplayStepKind.ENTROPY)

    expect(entropy).toHaveLength(7)
    const first = entropy[0]
    const last = entropy.at(-1)
    expect(first?.kind === ReplayStepKind.ENTROPY && first.runningSum).toBeCloseTo(
      -0.3 * Math.log(0.3),
    )
    expect(last?.kind === ReplayStepKind.ENTROPY && last.runningSum).toBeCloseTo(
      DIVERGENT_SCORE.semantic_entropy,
      12,
    )
  })

  it("sums each meaning cluster's likelihood weight for the likelihood-weighted estimator", () => {
    const replay = divergentReplay()

    expect(replay.clusterWeights?.[0]).toBeCloseTo(3 * Math.exp(-0.4))
    expect(replay.clusterWeights?.[1]).toBeCloseTo(2 * Math.exp(-0.4))
  })

  it('weights each meaning cluster by its own answers, not by its size', () => {
    const replay = buildReplay(
      WEIGHTED_TRACE,
      WEIGHTED_SCORE_TRACED.clusters,
      WEIGHTED_SCORE_TRACED.semantic_entropy,
    )

    // |c| / N would be 2/3 and 1/3; the lone "Sydney" carries 0.6 of the likelihood.
    expect(replay.clusterWeights?.[0]).toBeCloseTo(0.4, 12)
    expect(replay.clusterWeights?.[1]).toBeCloseTo(0.6, 12)
    expect(replay.probabilities).toEqual([0.4, 0.6])
  })

  it('refuses a likelihood-weighted p(c) that is not the weight share', () => {
    // These are |c| / N, the discrete p(c), under the likelihood-weighted estimator.
    const trace = { ...WEIGHTED_TRACE, cluster_probabilities: [2 / 3, 1 / 3] }
    const se = -(2 / 3) * Math.log(2 / 3) - (1 / 3) * Math.log(1 / 3)

    expect(() => buildReplay(trace, WEIGHTED_SCORE_TRACED.clusters, se)).toThrow(
      /p\(c\) 0\.667 for meaning cluster 1, but its estimator gives 0\.400/,
    )
  })

  it('refuses a discrete p(c) that is not |c| / N', () => {
    const trace = { ...TYPED_TRACE, cluster_probabilities: [0.4, 0.6] }
    const se = -0.4 * Math.log(0.4) - 0.6 * Math.log(0.6)

    expect(() => buildReplay(trace, TYPED_SCORE_TRACED.clusters, se)).toThrow(
      /p\(c\) 0\.400 for meaning cluster 1, but its estimator gives 0\.500/,
    )
  })

  it('refuses likelihoods that do not match the estimator', () => {
    const { clusters, semantic_entropy } = TYPED_SCORE_TRACED
    const weightedWithout = { ...WEIGHTED_TRACE, answer_likelihoods: null }
    const discreteWith = { ...TYPED_TRACE, answer_likelihoods: [0.5, 0.5] }

    expect(() =>
      buildReplay(
        weightedWithout,
        WEIGHTED_SCORE_TRACED.clusters,
        WEIGHTED_SCORE_TRACED.semantic_entropy,
      ),
    ).toThrow(/no answer likelihoods for the likelihood-weighted estimator/)
    expect(() => buildReplay(discreteWith, clusters, semantic_entropy)).toThrow(
      /answer likelihoods the discrete estimator does not use/,
    )
  })

  it('has no weights for the discrete estimator', () => {
    const replay = buildReplay(
      TYPED_TRACE,
      TYPED_SCORE_TRACED.clusters,
      TYPED_SCORE_TRACED.semantic_entropy,
    )

    expect(replay.clusterWeights).toBeNull()
    expect(replay.steps).toHaveLength(1 + 1 + 2)
  })

  it('refuses a trace that does not rebuild the returned clusters', () => {
    const clusters = [
      { member_indices: [0, 3], representative_text: 'Sally Ride' },
      ...DIVERGENT_SCORE.clusters.slice(1),
    ]

    expect(() => buildReplay(DIVERGENT_TRACE, clusters, DIVERGENT_SCORE.semantic_entropy)).toThrow(
      /does not rebuild the returned clusters/,
    )
  })

  it('refuses a trace whose p(c) does not give the reported SE', () => {
    expect(() => buildReplay(DIVERGENT_TRACE, DIVERGENT_SCORE.clusters, 1.0)).toThrow(
      /gives SE 1\.834.*reported 1\.000/,
    )
  })

  it('refuses a comparison against anything but the meaning cluster representative', () => {
    const comparisons = DIVERGENT_TRACE.clustering.comparisons.map((c, i) =>
      i === 0 ? { ...c, representative_index: 5 } : c,
    )
    const trace = { ...DIVERGENT_TRACE, clustering: { ...DIVERGENT_TRACE.clustering, comparisons } }

    expect(() =>
      buildReplay(trace, DIVERGENT_SCORE.clusters, DIVERGENT_SCORE.semantic_entropy),
    ).toThrow(/representative/)
  })

  // Comparison 0 is answer 2 vs answer 1: forward contradiction, backward not run.
  function withFirstComparison(edit: Partial<ClusteringComparison>) {
    const comparisons = DIVERGENT_TRACE.clustering.comparisons.map((c, i) =>
      i === 0 ? { ...c, ...edit } : c,
    )
    const trace = { ...DIVERGENT_TRACE, clustering: { ...DIVERGENT_TRACE.clustering, comparisons } }
    return () => buildReplay(trace, DIVERGENT_SCORE.clusters, DIVERGENT_SCORE.semantic_entropy)
  }

  it('refuses a join its verdicts contradict', () => {
    expect(DIVERGENT_TRACE.clustering.comparisons[0]?.forward).toBe(NliLabel.CONTRADICTION)
    expect(withFirstComparison({ joined: true })).toThrow(/join its verdicts contradict/)
  })

  it('refuses a backward verdict after a forward non-entailment', () => {
    expect(withFirstComparison({ backward: NliLabel.ENTAILMENT })).toThrow(
      /backward check after a forward non-entailment/,
    )
  })

  it('refuses a forward entailment with no backward verdict', () => {
    expect(withFirstComparison({ forward: NliLabel.ENTAILMENT })).toThrow(
      /skips the backward check/,
    )
  })
})

/** A captured backend response, replayed as the UI would. */
function replayCaptured({ response }: CapturedScore) {
  if (!response.se_trace) throw new Error('the captured response carries no se_trace')
  return buildReplay(response.se_trace, response.clusters, response.semantic_entropy)
}

describe('buildReplay on real /api/score responses', () => {
  it('replays the paraphrased preset: ten answers, nine joins, one meaning cluster, SE 0', () => {
    const replay = replayCaptured(PARAPHRASED_PRESET)

    const compares = replay.steps.filter((s) => s.kind === ReplayStepKind.COMPARE)
    expect(compares).toHaveLength(9)
    expect(compares.every((s) => s.outcome === CompareOutcome.JOINED)).toBe(true)
    expect(membersOf(replay.steps.at(-1))).toEqual([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    const entropy = replay.steps.at(-1)
    expect(entropy?.kind === ReplayStepKind.ENTROPY && entropy.runningSum).toBe(0)
  })

  it('replays the divergent preset into the seven returned meaning clusters and its SE', () => {
    const { response } = DIVERGENT_PRESET
    const replay = replayCaptured(DIVERGENT_PRESET)

    expect(replay.steps).toHaveLength(1 + 25 + 7)
    expect(membersOf(replay.steps.at(-1))).toEqual(response.clusters.map((c) => c.member_indices))
    const entropy = replay.steps.at(-1)
    expect(entropy?.kind === ReplayStepKind.ENTROPY && entropy.runningSum).toBeCloseTo(
      response.semantic_entropy,
      12,
    )
  })

  it('has the hand-built divergent fixture differ from the real run only in its two edited verdicts', () => {
    const real = DIVERGENT_PRESET.response.se_trace?.clustering
    if (!real) throw new Error('the captured divergent response carries no se_trace')
    const differing = DIVERGENT_TRACE.clustering.comparisons.filter(
      (c, i) => JSON.stringify(c) !== JSON.stringify(real.comparisons[i]),
    )

    expect(DIVERGENT_TRACE.clustering.placements).toEqual(real.placements)
    expect(DIVERGENT_TRACE.clustering.comparisons).toHaveLength(real.comparisons.length)
    expect(differing.map((c) => [c.answer_index, c.cluster_index])).toEqual([
      [6, 0],
      [8, 0],
    ])
  })
})

describe('nliInput', () => {
  it('puts the question in front of the answer, as the real clustering does', () => {
    expect(nliInput('Capital of France?', 'Paris')).toBe('Capital of France? Paris')
  })
})
