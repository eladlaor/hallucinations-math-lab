import type { ComponentProps } from 'react'

import { cn } from '@/lib/cn'

/** An error from the backend, shown verbatim. */
export function ErrorAlert({ title, message, className, ...props }: ComponentProps<'div'> & {
  title: string
  message: string
}) {
  return (
    <div
      role="alert"
      className={cn(
        'rounded-lg border border-destructive/40 bg-destructive-surface p-4 text-sm',
        className,
      )}
      {...props}
    >
      <p className="font-semibold text-destructive">{title}</p>
      <pre className="mt-2 whitespace-pre-wrap break-words font-mono text-xs">{message}</pre>
    </div>
  )
}
