import { useEffect, useState } from 'react'

import { ErrorAlert } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardDescription, CardTitle } from '@/components/ui/card'
import { ComparisonView, RunReportView } from '@/features/results/ReportViews'
import { api, type ResultsListing, type ResultsReport } from '@/lib/api'
import { BENCHMARK_COMMAND, RequestState, ResultsKind } from '@/lib/constants'
import { useLifecycleLog } from '@/lib/hooks'
import { logger } from '@/lib/logger'

const COMPONENT = 'ResultsTab'

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export function ResultsTab() {
  useLifecycleLog(COMPONENT)
  const [listing, setListing] = useState<ResultsListing | null>(null)
  const [listingError, setListingError] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [reportState, setReportState] = useState<RequestState>(RequestState.IDLE)
  const [report, setReport] = useState<ResultsReport | null>(null)
  const [reportError, setReportError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .results()
      .then((loaded) => {
        if (!cancelled) setListing(loaded)
      })
      .catch((e: unknown) => {
        logger.error('results listing failed', { component: COMPONENT, error: e })
        if (!cancelled) setListingError(errorMessage(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function open(path: string) {
    logger.info('results entry opened', { component: COMPONENT, path })
    setSelected(path)
    setReport(null)
    setReportError(null)
    setReportState(RequestState.LOADING)
    try {
      setReport(await api.resultsReport(path))
      setReportState(RequestState.SUCCESS)
    } catch (e) {
      logger.error('results report failed', { component: COMPONENT, path, error: e })
      setReportError(errorMessage(e))
      setReportState(RequestState.ERROR)
    }
  }

  return (
    <div className="space-y-6">
      <p className="text-sm leading-relaxed text-muted-foreground">
        Results of <code className="font-mono text-xs">eval.run_benchmark</code>: how well each
        detector&apos;s score separated the model&apos;s right answers from its wrong ones. This tab
        only reads results; run benchmarks from the command line.
      </p>

      {listingError && <ErrorAlert title="Could not list the results" message={listingError} />}
      {!listing && !listingError && (
        <p role="status" className="text-sm text-muted-foreground">
          Looking for results…
        </p>
      )}

      {listing && listing.entries.length === 0 && (
        <Card className="space-y-3">
          <CardTitle>No benchmark results yet</CardTitle>
          <CardDescription>
            Nothing in <span className="font-mono text-xs">{listing.results_dir}</span>. Produce a
            results file (it needs the model — see the README), then reload this page:
          </CardDescription>
          <pre className="overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs">
            {BENCHMARK_COMMAND}
          </pre>
        </Card>
      )}

      {listing && listing.entries.length > 0 && (
        <nav aria-label="Results files" className="space-y-2">
          <p className="text-sm font-medium">Results files</p>
          <ul className="flex flex-wrap gap-2">
            {listing.entries.map((entry) => (
              <li key={entry.path}>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  aria-pressed={selected === entry.path}
                  className="font-mono aria-pressed:border-primary aria-pressed:text-primary"
                  onClick={() => void open(entry.path)}
                >
                  {entry.kind === ResultsKind.SWEEP
                    ? `${entry.path}/ — sweep, ${entry.n_files} models`
                    : entry.path}
                </Button>
              </li>
            ))}
          </ul>
        </nav>
      )}

      {reportState === RequestState.LOADING && (
        <p role="status" className="text-sm text-muted-foreground">
          Computing AUROC with bootstrap confidence intervals…
        </p>
      )}
      {reportError && <ErrorAlert title="Could not report on this file" message={reportError} />}
      {report?.kind === ResultsKind.RUN && <RunReportView path={report.path} report={report.report} />}
      {report?.kind === ResultsKind.SWEEP && (
        <ComparisonView path={report.path} report={report.report} />
      )}
    </div>
  )
}
