/**
 * Turns the backend's SE trace into the replay's steps: one per NLI comparison
 * the real clustering ran, then one per meaning cluster as -Σ p(c) ln p(c)
 * fills in. Nothing is re-simulated: the steps are the trace, in order, with
 * the meaning clusters as they stood after each one.
 *
 * The trace is checked as it is read (fail fast): every comparison must be
 * against the representative (first answer) of the next meaning cluster in
 * order, the clusters it builds must be exactly the ones the API returned, and
 * its p(c) must be the one its estimator implies (|c| / N for discrete SE, the
 * meaning cluster's share of the summed answer likelihoods for likelihood-weighted
 * SE) and must give the SE the API reported.
 */

import type { Cluster, ClusteringComparison, SemanticEntropyTrace } from '@/lib/api'
import { NliLabel, REPLAY_SE_TOLERANCE, SeEstimator } from '@/lib/constants'
import { formatScore } from '@/lib/scores'

export const ReplayStepKind = { FIRST: 'first', COMPARE: 'compare', ENTROPY: 'entropy' } as const
export type ReplayStepKind = (typeof ReplayStepKind)[keyof typeof ReplayStepKind]

/** What a comparison led to: a join, the next meaning cluster, or (after the last) a new one. */
export const CompareOutcome = { JOINED: 'joined', NEXT: 'next', OPENED: 'opened' } as const
export type CompareOutcome = (typeof CompareOutcome)[keyof typeof CompareOutcome]

interface StepBase {
  /** Member answer indices per meaning cluster, after this step. */
  clustersSoFar: number[][]
}

/** The first answer opens meaning cluster 1: there is nothing to compare it with. */
export interface FirstStep extends StepBase {
  kind: typeof ReplayStepKind.FIRST
  answerIndex: number
}

export interface CompareStep extends StepBase {
  kind: typeof ReplayStepKind.COMPARE
  comparison: ClusteringComparison
  outcome: CompareOutcome
}

/** One more meaning cluster's -p ln p term added to SE. */
export interface EntropyStep extends StepBase {
  kind: typeof ReplayStepKind.ENTROPY
  clusterIndex: number
  runningSum: number
}

export type ReplayStep = FirstStep | CompareStep | EntropyStep

export interface Replay {
  steps: ReplayStep[]
  estimator: SeEstimator
  probabilities: number[]
  /** -p ln p per meaning cluster. */
  terms: number[]
  /** exp(mean token logprob) per answer; null for the discrete estimator. */
  answerLikelihoods: number[] | null
  /** Each meaning cluster's summed answer likelihood; null for the discrete estimator. */
  clusterWeights: number[] | null
  semanticEntropy: number
}

/** The NLI input for one answer: the question in front, as SE_NLI_INPUT_TEMPLATE has it. */
export function nliInput(question: string, answer: string): string {
  return `${question} ${answer}`
}

function copyClusters(clusters: number[][]): number[][] {
  return clusters.map((members) => [...members])
}

function invalidTrace(message: string, context: Record<string, unknown>): Error {
  return new Error(`The SE trace ${message} (${JSON.stringify(context)}).`)
}

/** The clustering steps, rebuilt from the comparisons and checked against the placements. */
function clusteringSteps(trace: SemanticEntropyTrace): { steps: ReplayStep[]; built: number[][] } {
  const { comparisons, placements } = trace.clustering
  const steps: ReplayStep[] = []
  const built: number[][] = []
  let next = 0
  for (const [answerIndex, placement] of placements.entries()) {
    if (placement.answer_index !== answerIndex) {
      throw invalidTrace('has placements out of answer order', { answerIndex, placement })
    }
    if (answerIndex === 0) {
      built.push([0])
      steps.push({ kind: ReplayStepKind.FIRST, answerIndex, clustersSoFar: copyClusters(built) })
      continue
    }
    // Every cluster that existed before this answer, in order, until one joins.
    const nExisting = built.length
    let joinedCluster: number | null = null
    let clusterIndex = 0
    while (joinedCluster === null && clusterIndex < nExisting) {
      const comparison = comparisons[next]
      const representative = built[clusterIndex]?.[0]
      const context = { answerIndex, clusterIndex, comparison }
      if (comparison?.answer_index !== answerIndex || comparison.cluster_index !== clusterIndex) {
        throw invalidTrace('skips a comparison the clustering must have run', context)
      }
      if (comparison.representative_index !== representative) {
        throw invalidTrace(
          `compares against answer ${comparison.representative_index}, not the representative ${representative}`,
          context,
        )
      }
      // The backward check runs only after a forward entailment; joining needs both.
      const forwardEntails = comparison.forward === NliLabel.ENTAILMENT
      if ((comparison.backward === null) !== !forwardEntails) {
        throw invalidTrace(
          forwardEntails
            ? 'skips the backward check after a forward entailment'
            : 'runs a backward check after a forward non-entailment',
          context,
        )
      }
      if (comparison.joined !== (forwardEntails && comparison.backward === NliLabel.ENTAILMENT)) {
        throw invalidTrace('has a join its verdicts contradict', context)
      }
      next += 1
      const last = clusterIndex === nExisting - 1
      let outcome: CompareOutcome = last ? CompareOutcome.OPENED : CompareOutcome.NEXT
      if (comparison.joined) {
        built[clusterIndex]?.push(answerIndex)
        joinedCluster = clusterIndex
        outcome = CompareOutcome.JOINED
      } else if (last) {
        built.push([answerIndex])
      }
      steps.push({
        kind: ReplayStepKind.COMPARE,
        comparison,
        outcome,
        clustersSoFar: copyClusters(built),
      })
      clusterIndex += 1
    }
    const opened = joinedCluster === null
    const landedIn = opened ? built.length - 1 : joinedCluster
    if (placement.opened_new_cluster !== opened || placement.cluster_index !== landedIn) {
      throw invalidTrace('has a placement its comparisons contradict', { placement, landedIn })
    }
  }
  if (next !== comparisons.length) {
    throw invalidTrace('has comparisons after the last answer was placed', {
      used: next,
      total: comparisons.length,
    })
  }
  return { steps, built }
}

function sameClusters(built: number[][], clusters: Cluster[]): boolean {
  return (
    built.length === clusters.length &&
    built.every((members, i) => members.join(',') === clusters[i]?.member_indices.join(','))
  )
}

/**
 * Each meaning cluster's summed answer likelihood for the likelihood-weighted
 * estimator, null for the discrete one. Throws unless the trace carries
 * likelihoods exactly when its estimator uses them, one per answer.
 */
function clusterWeightsFor(trace: SemanticEntropyTrace, built: number[][]): number[] | null {
  const likelihoods = trace.answer_likelihoods
  const weighted = trace.estimator === SeEstimator.LIKELIHOOD_WEIGHTED
  if ((likelihoods !== null) !== weighted) {
    throw invalidTrace(
      weighted
        ? 'has no answer likelihoods for the likelihood-weighted estimator'
        : 'has answer likelihoods the discrete estimator does not use',
      { estimator: trace.estimator },
    )
  }
  if (likelihoods === null) return null
  const nAnswers = trace.clustering.placements.length
  if (likelihoods.length !== nAnswers) {
    throw invalidTrace('has a likelihood count that differs from the answers', {
      nLikelihoods: likelihoods.length,
      nAnswers,
    })
  }
  return built.map((members) =>
    members.reduce((sum, i) => sum + (likelihoods[i] ?? Number.NaN), 0),
  )
}

/** Throws unless every p(c) is the one the estimator implies from the clusters (and weights). */
function checkProbabilities(
  trace: SemanticEntropyTrace,
  built: number[][],
  clusterWeights: number[] | null,
): void {
  const nAnswers = trace.clustering.placements.length
  const totalWeight = clusterWeights?.reduce((sum, w) => sum + w, 0) ?? Number.NaN
  for (const [clusterIndex, members] of built.entries()) {
    const reported = trace.cluster_probabilities[clusterIndex] ?? Number.NaN
    const implied =
      clusterWeights === null
        ? members.length / nAnswers
        : (clusterWeights[clusterIndex] ?? Number.NaN) / totalWeight
    // Written so that a NaN on either side fails the check.
    if (!(Math.abs(implied - reported) <= REPLAY_SE_TOLERANCE)) {
      throw invalidTrace(
        `has p(c) ${formatScore(reported)} for meaning cluster ${clusterIndex + 1}, but its estimator gives ${formatScore(implied)}`,
        { estimator: trace.estimator, clusterIndex, members },
      )
    }
  }
}

function entropyTerm(p: number): number {
  return p > 0 ? -p * Math.log(p) : 0
}

/** The replay of one traced SE run. Throws if the trace does not match the clusters or the SE. */
export function buildReplay(
  trace: SemanticEntropyTrace,
  clusters: Cluster[],
  semanticEntropy: number,
): Replay {
  const { steps, built } = clusteringSteps(trace)
  if (!sameClusters(built, clusters)) {
    throw invalidTrace('does not rebuild the returned clusters', {
      rebuilt: built,
      returned: clusters.map((c) => c.member_indices),
    })
  }
  const probabilities = trace.cluster_probabilities
  if (probabilities.length !== clusters.length) {
    throw invalidTrace('has a p(c) per cluster count that differs from the clusters', {
      nProbabilities: probabilities.length,
      nClusters: clusters.length,
    })
  }
  const clusterWeights = clusterWeightsFor(trace, built)
  checkProbabilities(trace, built, clusterWeights)
  const terms = probabilities.map(entropyTerm)
  let runningSum = 0
  for (const [clusterIndex, term] of terms.entries()) {
    runningSum += term
    steps.push({
      kind: ReplayStepKind.ENTROPY,
      clusterIndex,
      runningSum,
      clustersSoFar: copyClusters(built),
    })
  }
  if (!Number.isFinite(semanticEntropy)) {
    throw invalidTrace(`has no reported SE to end on (got ${semanticEntropy})`, {
      estimator: trace.estimator,
    })
  }
  if (Math.abs(runningSum - semanticEntropy) > REPLAY_SE_TOLERANCE) {
    throw invalidTrace(
      `gives SE ${formatScore(runningSum)} but the response reported ${formatScore(semanticEntropy)}`,
      { estimator: trace.estimator },
    )
  }
  return {
    steps,
    estimator: trace.estimator,
    probabilities,
    terms,
    answerLikelihoods: trace.answer_likelihoods,
    clusterWeights,
    semanticEntropy,
  }
}
