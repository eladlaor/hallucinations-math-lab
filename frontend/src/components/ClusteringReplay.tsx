import { useId, useMemo, useState } from 'react'

import { ClustersSoFar, CurrentStep, EntropyPanel } from '@/components/ReplaySteps'
import { ErrorAlert } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardDescription, CardTitle } from '@/components/ui/card'
import type { Cluster, SemanticEntropyTrace } from '@/lib/api'
import {
  DEFAULT_REPLAY_SPEED,
  REPLAY_SPEED_LABEL,
  REPLAY_STEP_MS,
  ReplaySpeed,
} from '@/lib/constants'
import { useLifecycleLog, useStepPlayback } from '@/lib/hooks'
import { logger } from '@/lib/logger'
import { buildReplay, ReplayStepKind, type Replay } from '@/lib/replay'

const COMPONENT = 'ClusteringReplay'

type Built = { replay: Replay; error: null } | { replay: null; error: string }

function build(
  trace: SemanticEntropyTrace,
  clusters: Cluster[],
  semanticEntropy: number,
): Built {
  try {
    return { replay: buildReplay(trace, clusters, semanticEntropy), error: null }
  } catch (e) {
    const error = e instanceof Error ? e.message : String(e)
    logger.error('replay could not be built', { component: COMPONENT, error })
    return { replay: null, error }
  }
}

/**
 * A step-through replay of the real SE clustering run (the API's `se_trace`):
 * each NLI comparison in the order it ran, then SE filling in from p(c).
 * `typedAnswers`: some answers were typed or edited (the Score tab), so they
 * carry no token logprobs; the SE step then says why SE fell back to discrete.
 */
export function ClusteringReplay({
  question,
  answers,
  clusters,
  trace,
  semanticEntropy,
  typedAnswers,
}: {
  question: string
  answers: string[]
  clusters: Cluster[]
  trace: SemanticEntropyTrace
  semanticEntropy: number
  typedAnswers: boolean
}) {
  const built = useMemo(
    () => build(trace, clusters, semanticEntropy),
    [trace, clusters, semanticEntropy],
  )
  return (
    <Card aria-label="Clustering replay" className="space-y-4">
      <div className="space-y-1">
        <CardTitle>Replay: how the answers split into meaning clusters</CardTitle>
        <CardDescription>
          DeBERTa compares each answer with the first answer of each meaning cluster, in order,
          with the question in front of both: first, does that first answer entail this one; only if
          so, the reverse. The answer joins the first meaning cluster where both directions are
          entailment and stops looking; if none match, it opens a new one. Then SE is computed from
          the meaning clusters. These are the comparisons the real run made.
        </CardDescription>
      </div>
      {built.replay ? (
        <ReplayPlayer
          replay={built.replay}
          question={question}
          answers={answers}
          typedAnswers={typedAnswers}
        />
      ) : (
        <ErrorAlert title="The replay could not be built" message={built.error} />
      )}
    </Card>
  )
}

function ReplayPlayer({
  replay,
  question,
  answers,
  typedAnswers,
}: {
  replay: Replay
  question: string
  answers: string[]
  typedAnswers: boolean
}) {
  const nSteps = replay.steps.length
  useLifecycleLog(COMPONENT, { nSteps, estimator: replay.estimator })
  const speedId = useId()
  const [speed, setSpeed] = useState<ReplaySpeed>(DEFAULT_REPLAY_SPEED)
  const playback = useStepPlayback(nSteps, REPLAY_STEP_MS[speed])
  const step = replay.steps[playback.index]
  if (!step) throw new Error(`Replay step ${playback.index} out of range 0..${nSteps - 1}.`)
  const activeAnswer =
    step.kind === ReplayStepKind.FIRST
      ? step.answerIndex
      : step.kind === ReplayStepKind.COMPARE
        ? step.comparison.answer_index
        : null

  function act(action: string, run: () => void) {
    logger.info('replay control used', {
      component: COMPONENT,
      action,
      step: playback.index + 1,
      nSteps,
      speed,
    })
    run()
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        {/* aria-disabled, not disabled, at the two ends: a natively disabled button
            drops keyboard focus to the page body the moment it is reached. */}
        <Button
          type="button"
          variant="outline"
          size="sm"
          aria-disabled={playback.atStart}
          onClick={() => {
            if (!playback.atStart) act('back', playback.back)
          }}
        >
          Step back
        </Button>
        {playback.playing ? (
          <Button type="button" size="sm" onClick={() => act('pause', playback.pause)}>
            Pause
          </Button>
        ) : (
          <Button type="button" size="sm" onClick={() => act('play', playback.play)}>
            Play
          </Button>
        )}
        <Button
          type="button"
          variant="outline"
          size="sm"
          aria-disabled={playback.atEnd}
          onClick={() => {
            if (!playback.atEnd) act('next', playback.next)
          }}
        >
          Next step
        </Button>
        <label htmlFor={speedId} className="ml-2 text-xs font-medium text-muted-foreground">
          Speed
        </label>
        <select
          id={speedId}
          value={speed}
          onChange={(e) => {
            const chosen = e.target.value as ReplaySpeed
            act('speed', () => setSpeed(chosen))
          }}
          className="h-8 rounded-md border border-input bg-card px-2 text-xs"
        >
          {Object.values(ReplaySpeed).map((value) => (
            <option key={value} value={value}>
              {REPLAY_SPEED_LABEL[value]}
            </option>
          ))}
        </select>
        {/* Not a live region: during playback it would be announced up to 4 times a second. */}
        <p className="ml-auto text-xs tabular-nums text-muted-foreground">
          Step {playback.index + 1} of {nSteps}
        </p>
      </div>

      <CurrentStep
        step={step}
        replay={replay}
        question={question}
        answers={answers}
        announce={!playback.playing}
      />
      <ClustersSoFar clusters={step.clustersSoFar} answers={answers} activeAnswer={activeAnswer} />
      {step.kind === ReplayStepKind.ENTROPY && (
        <EntropyPanel
          step={step}
          replay={replay}
          clusters={step.clustersSoFar}
          typedAnswers={typedAnswers}
        />
      )}
    </div>
  )
}
