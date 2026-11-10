import { useEffect, useState } from 'react'

import { ELAPSED_TICK_MS } from '@/lib/constants'
import { logger, type LogContext } from '@/lib/logger'

/** Logs the component's mount and unmount with structured context. */
export function useLifecycleLog(component: string, context: LogContext = {}): void {
  const serialized = JSON.stringify(context)
  useEffect(() => {
    const ctx = JSON.parse(serialized) as LogContext
    logger.info('component mounted', { component, ...ctx })
    return () => logger.info('component unmounted', { component, ...ctx })
  }, [component, serialized])
}

export interface StepPlayback {
  index: number
  playing: boolean
  atStart: boolean
  atEnd: boolean
  /** Play from the current step, or from the first once at the end. */
  play: () => void
  pause: () => void
  /** A manual step pauses playback. */
  next: () => void
  /** A manual step pauses playback. */
  back: () => void
}

/**
 * A step position over `nSteps` steps that advances one step every `stepMs`
 * while playing, and pauses on the last step. A new `stepMs` restarts the
 * current step's timer at the new pace.
 */
export function useStepPlayback(nSteps: number, stepMs: number): StepPlayback {
  const [index, setIndex] = useState(0)
  const [wantsPlay, setWantsPlay] = useState(false)
  const last = nSteps - 1
  const atEnd = index >= last
  const playing = wantsPlay && !atEnd

  useEffect(() => {
    if (!playing) return
    const id = window.setTimeout(() => {
      const nextIndex = Math.min(index + 1, last)
      setIndex(nextIndex)
      // Reaching the end pauses, so a later Step back does not resume playing.
      if (nextIndex >= last) setWantsPlay(false)
    }, stepMs)
    return () => window.clearTimeout(id)
  }, [playing, index, stepMs, last])

  return {
    index,
    playing,
    atStart: index === 0,
    atEnd,
    play: () => {
      if (atEnd) setIndex(0)
      setWantsPlay(true)
    },
    pause: () => setWantsPlay(false),
    next: () => {
      setWantsPlay(false)
      setIndex((i) => Math.min(i + 1, last))
    },
    back: () => {
      setWantsPlay(false)
      setIndex((i) => Math.max(i - 1, 0))
    },
  }
}

/** Milliseconds since `running` became true; resets when it turns false. */
export function useElapsed(running: boolean): number {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    if (!running) return
    const started = Date.now()
    const id = window.setInterval(() => setElapsed(Date.now() - started), ELAPSED_TICK_MS)
    return () => {
      window.clearInterval(id)
      setElapsed(0)
    }
  }, [running])
  return elapsed
}
