import * as SliderPrimitive from '@radix-ui/react-slider'
import type { ComponentProps } from 'react'

import { cn } from '@/lib/cn'

/** A single-thumb slider. `thumbLabel` is the thumb's accessible name. */
export function Slider({
  className,
  thumbLabel,
  ...props
}: ComponentProps<typeof SliderPrimitive.Root> & { thumbLabel: string }) {
  return (
    <SliderPrimitive.Root
      className={cn('relative flex w-full touch-none select-none items-center py-2', className)}
      {...props}
    >
      <SliderPrimitive.Track className="relative h-1.5 w-full grow overflow-hidden rounded-full bg-muted">
        <SliderPrimitive.Range className="absolute h-full bg-primary" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        aria-label={thumbLabel}
        className="block size-4 rounded-full border-2 border-primary bg-card shadow transition-colors"
      />
    </SliderPrimitive.Root>
  )
}
