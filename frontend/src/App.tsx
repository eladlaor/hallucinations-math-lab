import { useEffect, useState } from 'react'

import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { AskTab } from '@/features/ask/AskTab'
import { ResultsTab } from '@/features/results/ResultsTab'
import { ScoreTab } from '@/features/score/ScoreTab'
import { api, type Status } from '@/lib/api'
import { TabId } from '@/lib/constants'
import { useLifecycleLog } from '@/lib/hooks'
import { logger } from '@/lib/logger'

const COMPONENT = 'App'
/** Used until /api/status answers; the server's SE_THRESHOLD replaces it. */
const FALLBACK_DISPLAY_THRESHOLD = 1.0

export default function App() {
  useLifecycleLog(COMPONENT)
  const [tab, setTab] = useState<string>(TabId.SCORE)
  const [status, setStatus] = useState<Status | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .status()
      .then((loaded) => {
        if (!cancelled) setStatus(loaded)
      })
      .catch((e: unknown) => {
        logger.error('status failed to load', { component: COMPONENT, error: e })
        if (!cancelled) setStatusError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="mx-auto max-w-3xl px-5 py-10 sm:py-14">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">hallucinations-math-lab</h1>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Ask a model the same question several times. If it knows the answer, the answers agree
          in meaning; if it is making something up, they scatter. Semantic Entropy turns that
          spread into a number.
        </p>
      </header>

      <main className="mt-8">
        <Tabs
          value={tab}
          onValueChange={(value) => {
            logger.info('tab selected', { component: COMPONENT, tab: value })
            setTab(value)
          }}
        >
          <TabsList aria-label="Ways in">
            <TabsTrigger value={TabId.SCORE}>Score answers</TabsTrigger>
            <TabsTrigger value={TabId.ASK}>Ask the model</TabsTrigger>
            <TabsTrigger value={TabId.RESULTS}>Measure the detectors</TabsTrigger>
          </TabsList>
          <TabsContent value={TabId.SCORE}>
            <ScoreTab
              defaultThreshold={status?.se_threshold ?? FALLBACK_DISPLAY_THRESHOLD}
              replayUnavailableReason={
                status === null ? undefined : status.se_trace_unavailable_reason
              }
            />
          </TabsContent>
          <TabsContent value={TabId.ASK}>
            <AskTab status={status} statusError={statusError} />
          </TabsContent>
          <TabsContent value={TabId.RESULTS}>
            <ResultsTab />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
