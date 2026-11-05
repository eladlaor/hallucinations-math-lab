import { Slider } from '@/components/ui/slider'
import { THRESHOLD_STEP_NATS } from '@/lib/constants'
import { logger } from '@/lib/logger'
import { formatScore, shouldEscalate } from '@/lib/scores'

/**
 * The escalation threshold. Escalate is just SE > threshold, recomputed in the
 * browser as the slider moves — no request. A tick on the track marks where
 * this SE sits.
 */
export function ThresholdControl({
  semanticEntropy,
  threshold,
  max,
  onChange,
}: {
  semanticEntropy: number
  threshold: number
  max: number
  onChange: (value: number) => void
}) {
  const escalate = shouldEscalate(semanticEntropy, threshold)
  const sePosition = `${Math.min(100, (semanticEntropy / max) * 100)}%`
  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between gap-4">
        <p className="text-sm font-medium">
          Escalation threshold
        </p>
        <p className="text-sm tabular-nums text-muted-foreground">
          {formatScore(threshold, 2)} nats
        </p>
      </div>
      <div className="relative">
        <Slider
          thumbLabel="Escalation threshold"
          min={0}
          max={max}
          step={THRESHOLD_STEP_NATS}
          value={[threshold]}
          onValueChange={([value]) => {
            if (value === undefined) return
            logger.debug('threshold moved', { threshold: value, escalate: value < semanticEntropy })
            onChange(value)
          }}
          onValueCommit={([value]) =>
            logger.info('threshold set', {
              threshold: value,
              semanticEntropy,
              escalate: value !== undefined && shouldEscalate(semanticEntropy, value),
            })
          }
        />
        <div
          className="pointer-events-none absolute -bottom-3 h-2 w-0.5 -translate-x-1/2 bg-foreground"
          style={{ left: sePosition }}
          aria-hidden="true"
          title={`SE ${formatScore(semanticEntropy)}`}
        />
      </div>
      <p className="pt-2 text-xs text-muted-foreground">
        The tick under the track marks this SE ({formatScore(semanticEntropy)}). Answers whose SE
        is above the threshold are handed to the stronger judge.
      </p>
      <div
        role="status"
        aria-label="Escalation decision"
        aria-live="polite"
        className={
          escalate
            ? 'rounded-md border border-warn/40 px-3 py-2 text-sm'
            : 'rounded-md border border-good/40 px-3 py-2 text-sm'
        }
      >
        {escalate ? (
          <>
            <span className="font-semibold text-warn">▲ Escalate</span> — SE{' '}
            {formatScore(semanticEntropy)} is above {formatScore(threshold, 2)}; the judge would
            answer instead.
          </>
        ) : (
          <>
            <span className="font-semibold text-good">✓ Don&apos;t escalate</span> — SE{' '}
            {formatScore(semanticEntropy)} is not above {formatScore(threshold, 2)}; keep the
            model&apos;s answer.
          </>
        )}
      </div>
    </div>
  )
}
