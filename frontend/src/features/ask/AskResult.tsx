import { ClusterGroups } from '@/components/ClusterGroups'
import { Metric } from '@/components/Metric'
import { Card, CardDescription, CardTitle } from '@/components/ui/card'
import type { DetectionResult } from '@/lib/api'
import { AnswerSource } from '@/lib/constants'
import { detectorInfo, formatScore, sortDetectors } from '@/lib/scores'

/** What `demo.cli detect` / `chat` print, laid out: answer, decision, scores, clusters. */
export function AskResult({ result }: { result: DetectionResult }) {
  const fromJudge = result.final_answer_source === AnswerSource.JUDGE_MODEL
  const escalationScore = result.detector_scores[result.escalation_detector]
  const escalationLabel = detectorInfo(result.escalation_detector).short
  return (
    <div className="space-y-6">
      <Card aria-label="Answer" className="space-y-3">
        <CardTitle>Answer</CardTitle>
        <p className="text-lg leading-snug">{result.final_answer}</p>
        <p className="text-xs text-muted-foreground">
          {fromJudge
            ? 'From the Gemini judge, because the question escalated.'
            : "The model's own greedy answer."}
        </p>
        {fromJudge && (
          <p className="text-sm">
            <span className="text-muted-foreground">The model said: </span>
            {result.greedy_answer}
          </p>
        )}
      </Card>

      <div
        role="status"
        aria-label="Escalation decision"
        className={
          result.escalated
            ? 'rounded-md border border-warn/40 px-3 py-2 text-sm'
            : 'rounded-md border border-good/40 px-3 py-2 text-sm'
        }
      >
        {result.escalated ? (
          <span className="font-semibold text-warn">▲ Escalated</span>
        ) : (
          <span className="font-semibold text-good">✓ Not escalated</span>
        )}{' '}
        — {escalationLabel}{' '}
        {escalationScore === undefined ? 'n/a' : formatScore(escalationScore)}{' '}
        {result.escalated ? 'is above' : 'is not above'} the threshold{' '}
        {formatScore(result.escalation_threshold, 2)}.
        {result.escalated && !fromJudge && ' Detect never calls the judge; Chat would.'}
      </div>

      <Card className="grid gap-6 sm:grid-cols-3" aria-label="Detector scores">
        {sortDetectors(Object.keys(result.detector_scores)).map((name) => {
          const info = detectorInfo(name)
          return (
            <Metric
              key={name}
              label={`${info.name} (${info.short})`}
              value={formatScore(result.detector_scores[name] ?? Number.NaN)}
              note={info.meaning}
            />
          )
        })}
      </Card>

      {result.judge_reasoning && (
        <Card aria-label="Judge reasoning" className="space-y-2">
          <CardTitle>Judge reasoning</CardTitle>
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{result.judge_reasoning}</p>
        </Card>
      )}

      {result.clusters && (
        <Card className="space-y-2">
          <CardTitle>The {result.samples.length} samples, grouped by meaning</CardTitle>
          <CardDescription>
            Each colour and number is one meaning. SE is the entropy of these group sizes.
          </CardDescription>
          <ClusterGroups answers={result.samples} clusters={result.clusters} />
        </Card>
      )}
    </div>
  )
}
