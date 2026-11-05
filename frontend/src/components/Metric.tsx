import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

/** One score with its plain-language "what this means" note. */
export function Metric({
  label,
  value,
  note,
  className,
}: {
  label: string
  value: ReactNode
  note: ReactNode
  className?: string
}) {
  return (
    <div role="group" aria-label={label} className={cn('min-w-0', className)}>
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums">{value}</p>
      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{note}</p>
    </div>
  )
}
