import type { Status } from '@/lib/api'

/** Which model the Ask tab will run, and whether Chat can reach the judge. */
export function StatusStrip({ status }: { status: Status }) {
  return (
    <section
      aria-label="Model status"
      className="flex flex-wrap gap-x-6 gap-y-1 rounded-lg border bg-card px-4 py-3 text-sm"
    >
      <p>
        <span className="text-muted-foreground">Provider </span>
        <span className="font-mono text-xs">{status.provider}</span>
      </p>
      <p>
        <span className="text-muted-foreground">Model </span>
        <span className="font-mono text-xs">{status.model_id}</span>
      </p>
      <p>
        <span className="text-muted-foreground">Judge </span>
        {status.judge_configured ? (
          <span className="text-good">✓ configured</span>
        ) : (
          <span className="text-warn">✕ not configured</span>
        )}
      </p>
    </section>
  )
}
