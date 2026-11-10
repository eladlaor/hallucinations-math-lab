import { useEffect, useId, useState, type FormEvent } from 'react'

import { ClusterGroups } from '@/components/ClusterGroups'
import { ClusteringReplay } from '@/components/ClusteringReplay'
import { Metric } from '@/components/Metric'
import { ErrorAlert } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardDescription, CardTitle } from '@/components/ui/card'
import { Input, Label } from '@/components/ui/input'
import { ThresholdControl } from '@/features/score/ThresholdControl'
import { api, type Preset, type ScoreAnswer, type ScoreResponse, type TokenLogprob } from '@/lib/api'
import { DetectorName, RequestState } from '@/lib/constants'
import { useElapsed, useLifecycleLog } from '@/lib/hooks'
import { logger } from '@/lib/logger'
import {
  detectorInfo,
  formatElapsed,
  formatScore,
  seEstimatorNote,
  thresholdSliderMax,
} from '@/lib/scores'

const COMPONENT = 'ScoreTab'

interface AnswerDraft {
  id: number
  text: string
  /** From a preset; dropped as soon as the text is edited, since it no longer matches. */
  tokenLogprobs: TokenLogprob[] | null
}

interface ScoredSnapshot {
  /** A fresh id per scoring, so a new result restarts the replay. */
  runId: number
  question: string
  answers: string[]
  /** Some answer was typed or edited, so it carries no token logprobs. */
  typedAnswers: boolean
  response: ScoreResponse
  /** Why this scoring did not ask for the replay; null when it did. */
  replaySkippedReason: string | null
}

let nextRunId = 0
let nextAnswerId = 0
function draft(text = '', tokenLogprobs: TokenLogprob[] | null = null): AnswerDraft {
  nextAnswerId += 1
  return { id: nextAnswerId, text, tokenLogprobs }
}

function toRequestAnswer(answer: AnswerDraft): ScoreAnswer {
  return answer.tokenLogprobs
    ? { text: answer.text, token_logprobs: answer.tokenLogprobs }
    : { text: answer.text }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** Why the replay is skipped while /api/status has not answered (loading or failed). */
const STATUS_UNKNOWN_REASON =
  'the server status has not loaded, so the replay was not requested.'

/**
 * `replayUnavailableReason` (from /api/status): null when the server can record
 * the SE trace, a string saying why it cannot, or undefined while the status is
 * unknown (still loading, or failed). The trace is asked for only on null: a
 * server that cannot record it answers a request for it with a 400.
 */
export function ScoreTab({
  defaultThreshold,
  replayUnavailableReason,
}: {
  defaultThreshold: number
  replayUnavailableReason: string | null | undefined
}) {
  useLifecycleLog(COMPONENT)
  const formId = useId()
  const [presets, setPresets] = useState<Preset[]>([])
  const [presetsError, setPresetsError] = useState<string | null>(null)
  const [activePreset, setActivePreset] = useState<string | null>(null)
  const [question, setQuestion] = useState('')
  const [answers, setAnswers] = useState<AnswerDraft[]>(() => [draft()])
  const [state, setState] = useState<RequestState>(RequestState.IDLE)
  const [error, setError] = useState<string | null>(null)
  const [scored, setScored] = useState<ScoredSnapshot | null>(null)
  // null until the user moves the slider: follow the configured SE_THRESHOLD.
  const [chosenThreshold, setThreshold] = useState<number | null>(null)
  const threshold = chosenThreshold ?? defaultThreshold
  const loading = state === RequestState.LOADING
  const elapsed = useElapsed(loading)

  useEffect(() => {
    let cancelled = false
    api
      .presets()
      .then((loaded) => {
        if (!cancelled) setPresets(loaded)
      })
      .catch((e: unknown) => {
        logger.error('presets failed to load', { component: COMPONENT, error: e })
        if (!cancelled) setPresetsError(errorMessage(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  function loadPreset(preset: Preset) {
    logger.info('preset loaded', {
      component: COMPONENT,
      preset: preset.name,
      nAnswers: preset.samples.length,
    })
    setActivePreset(preset.name)
    setQuestion(preset.question)
    setAnswers(preset.samples.map((s) => draft(s.text, s.token_logprobs)))
    setScored(null)
    setError(null)
  }

  function editAnswer(id: number, text: string) {
    setActivePreset(null)
    setAnswers((current) =>
      current.map((a) => (a.id === id ? { ...a, text, tokenLogprobs: null } : a)),
    )
  }

  function addAnswer() {
    logger.info('answer added', { component: COMPONENT, nAnswers: answers.length + 1 })
    setAnswers((current) => [...current, draft()])
  }

  function removeAnswer(id: number, position: number) {
    logger.info('answer removed', { component: COMPONENT, position, nAnswers: answers.length - 1 })
    setActivePreset(null)
    setAnswers((current) => current.filter((a) => a.id !== id))
  }

  async function score(event: FormEvent) {
    event.preventDefault()
    // The trace costs no extra NLI calls, so the replay is asked for whenever the
    // server's NLI scorer is known to be able to record it.
    const replaySkippedReason =
      replayUnavailableReason === undefined ? STATUS_UNKNOWN_REASON : replayUnavailableReason
    const request = {
      question: question.trim(),
      answers: answers.map(toRequestAnswer),
      se_trace: replaySkippedReason === null,
    }
    const withLogprobs = answers.every((a) => a.tokenLogprobs !== null)
    const context = {
      component: COMPONENT,
      preset: activePreset,
      nAnswers: request.answers.length,
      withLogprobs,
      seTrace: request.se_trace,
    }
    logger.info('score requested', context)
    setState(RequestState.LOADING)
    setError(null)
    try {
      const response = await api.score(request)
      nextRunId += 1
      setScored({
        runId: nextRunId,
        question: request.question,
        answers: request.answers.map((a) => a.text),
        typedAnswers: !withLogprobs,
        response,
        replaySkippedReason,
      })
      setState(RequestState.SUCCESS)
      logger.info('score shown', {
        ...context,
        semanticEntropy: response.semantic_entropy,
        nClusters: response.clusters.length,
        traced: response.se_trace !== undefined,
      })
    } catch (e) {
      logger.error('score failed', { ...context, error: e })
      setError(errorMessage(e))
      setState(RequestState.ERROR)
    }
  }

  const canScore =
    !loading && question.trim() !== '' && answers.length > 0 && answers.every((a) => a.text.trim())
  const se = detectorInfo(DetectorName.SEMANTIC_ENTROPY)
  const pe = detectorInfo(DetectorName.PREDICTIVE_ENTROPY)

  return (
    <div className="space-y-6">
      <p className="text-sm leading-relaxed text-muted-foreground">
        Paste N answers to one question — or pick a preset — and see whether they mean the same
        thing. No model generates anything here: DeBERTa only checks which answers agree.
      </p>

      <div className="space-y-2">
        <p className="text-sm font-medium">Presets</p>
        {presetsError ? (
          <ErrorAlert title="Could not load the presets" message={presetsError} />
        ) : (
          <div className="flex flex-wrap gap-2">
            {presets.map((preset) => (
              <Button
                key={preset.name}
                type="button"
                variant="outline"
                size="sm"
                aria-pressed={activePreset === preset.name}
                className="capitalize aria-pressed:border-primary aria-pressed:text-primary"
                onClick={() => loadPreset(preset)}
              >
                {preset.name}
              </Button>
            ))}
          </div>
        )}
      </div>

      <form id={formId} onSubmit={score} className="space-y-4">
        <div className="space-y-1.5">
          <Label htmlFor={`${formId}-question`}>Question</Label>
          <Input
            id={`${formId}-question`}
            value={question}
            onChange={(e) => {
              setActivePreset(null)
              setQuestion(e.target.value)
            }}
            placeholder="What is the capital of France?"
          />
        </div>

        <fieldset className="space-y-2">
          <legend className="text-sm font-medium">Answers</legend>
          <ol className="space-y-2">
            {answers.map((answer, i) => (
              <li key={answer.id} className="flex items-center gap-2">
                <span className="w-5 shrink-0 text-right text-xs tabular-nums text-subtle-foreground">
                  {i + 1}
                </span>
                <Input
                  aria-label={`Answer ${i + 1}`}
                  value={answer.text}
                  onChange={(e) => editAnswer(answer.id, e.target.value)}
                />
                <span
                  className="w-16 shrink-0 text-[11px] text-subtle-foreground"
                  title={
                    answer.tokenLogprobs
                      ? 'This answer carries token logprobs, so PE can use it.'
                      : undefined
                  }
                >
                  {answer.tokenLogprobs ? 'logprobs' : ''}
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  aria-label={`Remove answer ${i + 1}`}
                  disabled={answers.length === 1}
                  onClick={() => removeAnswer(answer.id, i + 1)}
                >
                  ×
                </Button>
              </li>
            ))}
          </ol>
          <Button type="button" variant="ghost" size="sm" onClick={addAnswer}>
            Add answer
          </Button>
        </fieldset>

        <div className="flex items-center gap-3">
          <Button type="submit" disabled={!canScore}>
            Score answers
          </Button>
        </div>
      </form>

      {loading && (
        <p role="status" aria-label="Scoring progress" className="text-sm text-muted-foreground">
          Scoring… {formatElapsed(elapsed)}. The first request loads DeBERTa (about 1.6 GB,
          downloaded once), which can take a minute or more; later requests take seconds.
        </p>
      )}

      {error && <ErrorAlert title="Scoring failed" message={error} />}

      {scored && (
        <Card className="space-y-6" aria-label="Scores">
          <div className="grid gap-6 sm:grid-cols-3">
            <Metric
              label={`${se.name} (${se.short})`}
              value={formatScore(scored.response.semantic_entropy)}
              note={`${se.meaning} ${seEstimatorNote(scored.response.se_estimator)}`}
            />
            <Metric
              label="Meaning clusters"
              value={scored.response.clusters.length}
              note={`Groups of answers that mean the same thing, out of ${scored.response.n_answers}.`}
            />
            <Metric
              label={`${pe.name} (${pe.short})`}
              value={
                scored.response.predictive_entropy === null ? (
                  <span className="text-base font-medium text-muted-foreground">
                    needs token logprobs
                  </span>
                ) : (
                  formatScore(scored.response.predictive_entropy)
                )
              }
              note={
                scored.response.predictive_entropy === null
                  ? 'PE averages the probabilities the model gave its own tokens. Typed or edited answers have none, so there is nothing to average.'
                  : pe.meaning
              }
            />
          </div>

          <ThresholdControl
            semanticEntropy={scored.response.semantic_entropy}
            threshold={threshold}
            max={thresholdSliderMax(scored.response.n_answers)}
            onChange={setThreshold}
          />

          <div className="space-y-2">
            <CardTitle>Answers grouped by meaning</CardTitle>
            <CardDescription>
              Each colour and number is one meaning cluster. SE is the entropy over these meaning
              clusters; the replay, when recorded, shows the p(c) it used for each.
            </CardDescription>
            <ClusterGroups answers={scored.answers} clusters={scored.response.clusters} />
          </div>
        </Card>
      )}

      {scored?.response.se_trace && (
        <ClusteringReplay
          key={scored.runId}
          question={scored.question}
          answers={scored.answers}
          clusters={scored.response.clusters}
          trace={scored.response.se_trace}
          semanticEntropy={scored.response.semantic_entropy}
          typedAnswers={scored.typedAnswers}
        />
      )}

      {scored?.replaySkippedReason && (
        <p className="text-xs text-muted-foreground">
          No step-by-step replay: {scored.replaySkippedReason}
        </p>
      )}
    </div>
  )
}
