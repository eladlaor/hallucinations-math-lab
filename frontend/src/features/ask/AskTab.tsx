import { useId, useState } from 'react'

import { ErrorAlert } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input, Label } from '@/components/ui/input'
import { AskResult } from '@/features/ask/AskResult'
import { StatusStrip } from '@/features/ask/StatusStrip'
import { api, type DetectionResult, type Status } from '@/lib/api'
import { AskMode, README_ASK_URL, README_HARDWARE_URL, RequestState } from '@/lib/constants'
import { useElapsed, useLifecycleLog } from '@/lib/hooks'
import { logger } from '@/lib/logger'
import { formatElapsed } from '@/lib/scores'

const COMPONENT = 'AskTab'

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export function AskTab({ status, statusError }: { status: Status | null; statusError: string | null }) {
  useLifecycleLog(COMPONENT)
  const questionId = useId()
  const [question, setQuestion] = useState('')
  const [state, setState] = useState<RequestState>(RequestState.IDLE)
  const [runningMode, setRunningMode] = useState<AskMode | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  // On by default: the replay is the point of the playground. It is off and locked,
  // with the server's reason, when /ask cannot record it (no SE detector enabled, or
  // an NLI scorer without per-direction classify), so the default never 400s.
  const [wantsReplay, setWantsReplay] = useState(true)
  const replayUnavailableReason = status?.ask_se_trace_unavailable_reason ?? null
  const recordReplay = wantsReplay && replayUnavailableReason === null
  const loading = state === RequestState.LOADING
  const elapsed = useElapsed(loading)

  const ready = status !== null && !loading && question.trim() !== ''
  const chatBlocked = status !== null && !status.judge_configured

  async function run(mode: AskMode) {
    const context = { component: COMPONENT, mode, model: status?.model_id, seTrace: recordReplay }
    logger.info('ask requested', context)
    setState(RequestState.LOADING)
    setRunningMode(mode)
    setError(null)
    setResult(null)
    try {
      const response = await api.ask(question.trim(), mode, recordReplay)
      setResult(response)
      setState(RequestState.SUCCESS)
      logger.info('ask shown', {
        ...context,
        escalated: response.escalated,
        source: response.final_answer_source,
      })
    } catch (e) {
      logger.error('ask failed', { ...context, error: e })
      setError(errorMessage(e))
      setState(RequestState.ERROR)
    } finally {
      setRunningMode(null)
    }
  }

  return (
    <div className="space-y-6">
      <p className="text-sm leading-relaxed text-muted-foreground">
        Ask the model under test one question. It answers once greedily and samples several more
        answers; the detectors score how much those agree. <strong>Detect</strong> stops at the
        scores; <strong>Chat</strong> also hands an escalated question to the Gemini judge — like{' '}
        <a className="underline underline-offset-2" href={README_ASK_URL}>
          demo.cli detect / chat
        </a>
        .
      </p>

      {statusError && <ErrorAlert title="Could not read the server status" message={statusError} />}
      {status && <StatusStrip status={status} />}

      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault()
          if (ready) void run(AskMode.DETECT)
        }}
      >
        <div className="space-y-1.5">
          <Label htmlFor={questionId}>Question</Label>
          <Input
            id={questionId}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Who was the first woman to walk on Mars?"
          />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" disabled={!ready}>
            Detect
          </Button>
          <Button
            type="button"
            variant="outline"
            disabled={!ready || chatBlocked}
            aria-describedby={chatBlocked ? `${questionId}-chat-reason` : undefined}
            onClick={() => void run(AskMode.CHAT)}
          >
            Chat
          </Button>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={recordReplay}
            disabled={replayUnavailableReason !== null}
            aria-describedby={
              replayUnavailableReason !== null ? `${questionId}-replay-reason` : undefined
            }
            onChange={(e) => {
              logger.info('replay recording toggled', {
                component: COMPONENT,
                seTrace: e.target.checked,
              })
              setWantsReplay(e.target.checked)
            }}
          />
          Record the clustering for a step-by-step replay (no extra NLI calls)
        </label>
        {replayUnavailableReason !== null && (
          <p id={`${questionId}-replay-reason`} className="text-xs text-muted-foreground">
            The replay is unavailable: {replayUnavailableReason}
          </p>
        )}
        {chatBlocked && (
          <p id={`${questionId}-chat-reason`} className="text-xs text-muted-foreground">
            Chat is unavailable: {status.judge_unavailable_reason}
          </p>
        )}
        <p className="text-xs text-muted-foreground">
          The model runs on this machine or on the server in your .env. On a CPU one question can
          take minutes — see{' '}
          <a className="underline underline-offset-2" href={README_HARDWARE_URL}>
            what hardware you need
          </a>
          .
        </p>
      </form>

      {loading && (
        <p role="status" aria-label="Generation progress" className="text-sm text-muted-foreground">
          Generating answers with {status?.model_id ?? 'the model'}
          {runningMode === AskMode.CHAT ? ', then the judge if it escalates' : ''}…{' '}
          <span className="tabular-nums">{formatElapsed(elapsed)}</span> elapsed. The first
          question also loads the model.
        </p>
      )}

      {error && <ErrorAlert title="The model run failed" message={error} />}
      {result && <AskResult result={result} />}
    </div>
  )
}
