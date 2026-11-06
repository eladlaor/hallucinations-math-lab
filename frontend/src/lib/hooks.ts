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
