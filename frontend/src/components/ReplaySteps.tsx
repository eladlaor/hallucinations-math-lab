import type { ReactNode } from 'react'

import type { ClusteringComparison } from '@/lib/api'
import { clusterColour } from '@/lib/clusterPalette'
import { cn } from '@/lib/cn'
import { NliLabel, SeEstimator, type NliLabel as NliLabelValue } from '@/lib/constants'
import {
  CompareOutcome,
  nliInput,
  ReplayStepKind,
  type CompareStep,
  type EntropyStep,
  type Replay,
  type ReplayStep,
} from '@/lib/replay'
import { formatScore } from '@/lib/scores'

/** Answers are numbered from 1 on screen, as in the answer list. */
const shown = (index: number) => index + 1

function Quote({ children }: { children: ReactNode }) {
  return <q className="font-medium text-foreground">{children}</q>
}

function Swatch({ clusterIndex }: { clusterIndex: number }) {
  return (
    <span
      className="inline-block size-2.5 shrink-0 rounded-sm"
      style={{ background: clusterColour(clusterIndex) }}
      aria-hidden="true"
    />
  )
}

function VerdictBadge({ label }: { label: NliLabelValue }) {
  return (
    <span
      className={cn(
        'rounded px-1.5 py-0.5 text-xs font-semibold',
        label === NliLabel.ENTAILMENT ? 'text-good' : 'text-warn',
      )}
    >
      {label}
    </span>
  )
}

/** One direction: does the premise entail the hypothesis? Both carry the question in front. */
function Verdict({
  name,
  premise,
  hypothesis,
  label,
}: {
  name: string
  premise: string
  hypothesis: string
  label: NliLabelValue | null
}) {
  return (
    <div role="group" aria-label={name} className="space-y-1 rounded-md border px-3 py-2 text-sm">
      <p className="text-xs font-medium text-muted-foreground">{name}</p>
      <p className="leading-relaxed">
        Does <Quote>{premise}</Quote> entail <Quote>{hypothesis}</Quote>?{' '}
        {label === null ? (
          <span className="text-muted-foreground">
            Not run: the forward verdict was not entailment, so the pair cannot join.
          </span>
        ) : (
          <VerdictBadge label={label} />
        )}
      </p>
    </div>
  )
}

function outcomeText(step: CompareStep): string {
  const { answer_index: answer, cluster_index: cluster } = step.comparison
  switch (step.outcome) {
    case CompareOutcome.JOINED:
      return `Entailment both ways: answer ${shown(answer)} joins meaning cluster ${shown(cluster)}.`
    case CompareOutcome.NEXT:
      return `No match: on to meaning cluster ${shown(cluster + 1)}.`
    case CompareOutcome.OPENED:
      return `No match with any meaning cluster: answer ${shown(answer)} opens meaning cluster ${step.clustersSoFar.length}.`
  }
}

function Comparison({
  step,
  question,
  answers,
}: {
  step: CompareStep
  question: string
  answers: string[]
}) {
  const c: ClusteringComparison = step.comparison
  const answer = answers[c.answer_index] ?? ''
  const representative = answers[c.representative_index] ?? ''
  const answerInput = nliInput(question, answer)
  const representativeInput = nliInput(question, representative)
  return (
    <>
      <p>
        Answer {shown(c.answer_index)} <Quote>{answer}</Quote> is being placed. It is compared
        with meaning cluster {shown(c.cluster_index)}&apos;s representative, answer{' '}
        {shown(c.representative_index)} <Quote>{representative}</Quote> (the first answer of that
        meaning cluster).
      </p>
      <div className="grid gap-2 sm:grid-cols-2">
        <Verdict
          name="Forward verdict"
          premise={representativeInput}
          hypothesis={answerInput}
          label={c.forward}
        />
        <Verdict
          name="Backward verdict"
          premise={answerInput}
          hypothesis={representativeInput}
          label={c.backward}
        />
      </div>
      <p className="font-medium">{outcomeText(step)}</p>
    </>
  )
}

function EntropyTermLine({ step, replay }: { step: EntropyStep; replay: Replay }) {
  const p = replay.probabilities[step.clusterIndex] ?? 0
  const term = replay.terms[step.clusterIndex] ?? 0
  return (
    <p>
      Meaning cluster {shown(step.clusterIndex)} adds −p ln p = −{formatScore(p)} × ln{' '}
      {formatScore(p)} = {formatScore(term)} to SE.
    </p>
  )
}

/**
 * What happened in this step, in words. With `announce`, a screen reader reads
 * each new step; it is off during playback, which would flood it.
 */
export function CurrentStep({
  step,
  replay,
  question,
  answers,
  announce,
}: {
  step: ReplayStep
  replay: Replay
  question: string
  answers: string[]
  announce: boolean
}) {
  return (
    <section
      aria-label="Current step"
      aria-live={announce ? 'polite' : 'off'}
      aria-atomic="true"
      className="space-y-3 text-sm leading-relaxed"
    >
      {step.kind === ReplayStepKind.FIRST && (
        <p>
          Answer {shown(step.answerIndex)} <Quote>{answers[step.answerIndex]}</Quote> opens
          meaning cluster 1: there is no meaning cluster to compare it with yet.
        </p>
      )}
      {step.kind === ReplayStepKind.COMPARE && (
        <Comparison step={step} question={question} answers={answers} />
      )}
      {step.kind === ReplayStepKind.ENTROPY && <EntropyTermLine step={step} replay={replay} />}
    </section>
  )
}

/** The meaning clusters as they stand after this step; the answer being placed is ringed. */
export function ClustersSoFar({
  clusters,
  answers,
  activeAnswer,
}: {
  clusters: number[][]
  answers: string[]
  activeAnswer: number | null
}) {
  return (
    <section aria-label="Meaning clusters so far" className="space-y-2">
      <p className="text-xs font-medium text-muted-foreground">
        Meaning clusters so far: {clusters.length}
      </p>
      <ol className="space-y-2">
        {clusters.map((members, i) => (
          <li
            key={members[0]}
            aria-label={`Meaning cluster ${shown(i)}: ${members.length} ${members.length === 1 ? 'answer' : 'answers'}`}
            className="flex flex-wrap items-center gap-2 text-sm"
          >
            <Swatch clusterIndex={i} />
            <span className="text-xs text-muted-foreground">{shown(i)}</span>
            {members.map((answerIndex) => (
              <span
                key={answerIndex}
                className={cn(
                  'inline-flex max-w-full items-center gap-1.5 rounded-full border py-0.5 pl-1 pr-2.5',
                  answerIndex === activeAnswer && 'ring-2 ring-primary',
                )}
                style={{ borderColor: clusterColour(i) }}
              >
                <span className="text-[11px] tabular-nums text-subtle-foreground">
                  {shown(answerIndex)}
                </span>
                <span className="truncate">{answers[answerIndex]}</span>
              </span>
            ))}
          </li>
        ))}
      </ol>
    </section>
  )
}

function EstimatorNote({
  replay,
  nAnswers,
  typedAnswers,
}: {
  replay: Replay
  nAnswers: number
  typedAnswers: boolean
}) {
  if (replay.estimator === SeEstimator.DISCRETE) {
    return (
      <p>
        p(c) = |c| / N: each meaning cluster&apos;s share of the {nAnswers} answers, the discrete
        estimator.
        {typedAnswers && ' Typed or edited answers carry no token probabilities to weigh by.'}
      </p>
    )
  }
  return (
    <div className="space-y-1">
      <p>
        p(c) is each meaning cluster&apos;s share of the answers&apos; likelihoods. Each answer
        weighs exp(mean token logprob), its length-normalized sequence likelihood:
      </p>
      <p className="flex flex-wrap gap-x-3 text-xs tabular-nums text-muted-foreground">
        {(replay.answerLikelihoods ?? []).map((weight, i) => (
          <span key={i}>
            answer {shown(i)}: {formatScore(weight)}
          </span>
        ))}
      </p>
    </div>
  )
}

/**
 * SE = −Σ p(c) ln p(c), one meaning cluster's term at a time up to `step`.
 * `typedAnswers`: some answers carry no token logprobs, which is why the
 * discrete estimator ran; otherwise it ran because it was configured.
 */
export function EntropyPanel({
  step,
  replay,
  clusters,
  typedAnswers,
}: {
  step: EntropyStep
  replay: Replay
  clusters: number[][]
  typedAnswers: boolean
}) {
  const nAnswers = clusters.reduce((n, members) => n + members.length, 0)
  const weighted = replay.clusterWeights !== null
  const totalWeight = (replay.clusterWeights ?? []).reduce((sum, w) => sum + w, 0)
  const last = step.clusterIndex === clusters.length - 1
  return (
    <section aria-label="Semantic entropy" className="space-y-3 text-sm">
      <p className="font-semibold">SE = −Σ p(c) ln p(c)</p>
      <EstimatorNote replay={replay} nAnswers={nAnswers} typedAnswers={typedAnswers} />
      <table className="w-full text-left tabular-nums">
        <thead className="text-xs text-muted-foreground">
          <tr>
            <th className="py-1 font-medium">Meaning cluster</th>
            <th className="py-1 font-medium">{weighted ? 'Weight / total' : '|c| / N'}</th>
            <th className="py-1 font-medium">p(c)</th>
            <th className="py-1 font-medium">−p ln p</th>
          </tr>
        </thead>
        <tbody>
          {clusters.slice(0, step.clusterIndex + 1).map((members, i) => (
            <tr key={members[0]} aria-label={`Meaning cluster ${shown(i)}`}>
              <td className="py-1">
                <span className="inline-flex items-center gap-2">
                  <Swatch clusterIndex={i} />
                  {shown(i)}
                </span>
              </td>
              <td className="py-1">
                {weighted
                  ? `${formatScore(replay.clusterWeights?.[i] ?? 0)} / ${formatScore(totalWeight)}`
                  : `${members.length} / ${nAnswers}`}
              </td>
              <td className="py-1">{formatScore(replay.probabilities[i] ?? 0)}</td>
              <td className="py-1">{formatScore(replay.terms[i] ?? 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-base font-semibold tabular-nums">
        {last
          ? `SE = ${formatScore(step.runningSum)} nats`
          : `SE so far ${formatScore(step.runningSum)} nats`}
      </p>
    </section>
  )
}
