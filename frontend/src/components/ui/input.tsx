import type { ComponentProps } from 'react'

import { cn } from '@/lib/cn'

const FIELD =
  'w-full rounded-md border border-input bg-card px-3 py-2 text-sm placeholder:text-subtle-foreground disabled:opacity-50'

export function Input({ className, ...props }: ComponentProps<'input'>) {
  return <input className={cn(FIELD, 'h-9', className)} {...props} />
}

export function Label({ className, ...props }: ComponentProps<'label'>) {
  return <label className={cn('text-sm font-medium', className)} {...props} />
}
