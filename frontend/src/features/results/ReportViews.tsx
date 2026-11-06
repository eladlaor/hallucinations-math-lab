import { Card, CardDescription, CardTitle } from '@/components/ui/card'
import { AurocIntervalPlot } from '@/features/results/AurocIntervalPlot'
import type { AurocReport, ComparisonReport, Criterion } from '@/lib/api'
import { detectorInfo, formatScore, sortDetectors } from '@/lib/scores'

const CRITERION_LABELS: Record<string, string> = {
  se_auroc: 'SE AUROC reaches the target',
  se_minus_pe_auroc: "SE beats PE's AUROC by the target margin",
}

function CriterionLine({ id, criterion }: { id: string; criterion: Criterion }) {
  return (
    <li className="flex flex-wrap items-baseline gap-x-2 text-sm">
      {criterion.met ? (
        <span className="font-semibold text-good">✓ Met</span>
      ) : (
        <span className="font-semibold text-warn">✕ Not met</span>
      )}
      <span>{CRITERION_LABELS[id] ?? id}</span>
      <span className="tabular-nums text-muted-foreground">
        {formatScore(criterion.value)} [{formatScore(criterion.ci.low)},{' '}
        {formatScore(criterion.ci.high)}] vs target {criterion.target}
      </span>
    </li>
  )
}

/** One results file: run config, counts, AUROC per detector, success criteria. */
export function RunReportView({ path, report }: { path: string; report: AurocReport }) {
  const { run_config: config } = report
  return (
    <Card className="space-y-5" aria-label={`Report for ${path}`}>
      <div className="space-y-1">
        <CardTitle className="font-mono text-xs">{path}</CardTitle>
        <CardDescription>
          {config.model_id} · {config.sample_n} samples at temperature {config.sample_temperature}
        </CardDescription>
        <p className="text-sm">
          {report.n_rows} questions scored, {report.n_hallucinations} hallucinated (
          {formatScore((report.n_hallucinations / report.n_rows) * 100, 1)}%)
        </p>
      </div>
      <AurocIntervalPlot detectors={report.detectors} />
      {report.criteria && (
        <div className="space-y-2">
          <CardTitle>Success criteria</CardTitle>
          <ul className="space-y-1">
            {Object.entries(report.criteria).map(([id, criterion]) => (
              <CriterionLine key={id} id={id} criterion={criterion} />
            ))}
          </ul>
        </div>
      )}
    </Card>
  )
}

/** A sweep directory: the models side by side, as eval.compare_models prints them. */
export function ComparisonView({ path, report }: { path: string; report: ComparisonReport }) {
  const detectorNames = sortDetectors([
    ...new Set(report.models.flatMap((m) => Object.keys(m.detectors))),
  ])
  return (
    <Card className="space-y-4" aria-label={`Report for ${path}`}>
      <div className="space-y-1">
        <CardTitle className="font-mono text-xs">{path}/</CardTitle>
        <CardDescription>
          {report.n_questions} questions · {report.sampling.sample_n} samples at temperature{' '}
          {report.sampling.sample_temperature}
        </CardDescription>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <caption className="sr-only">Model comparison</caption>
          <thead className="text-xs text-muted-foreground">
            <tr className="border-b">
              <th scope="col" className="py-2 pr-4 font-medium">
                Model
              </th>
              <th scope="col" className="py-2 pr-4 text-right font-medium">
                Scored
              </th>
              <th scope="col" className="py-2 pr-4 text-right font-medium">
                Wrong
              </th>
              <th scope="col" className="py-2 pr-4 text-right font-medium">
                Accuracy
              </th>
              {detectorNames.map((name) => (
                <th key={name} scope="col" className="py-2 pr-4 text-right font-medium">
                  {detectorInfo(name).short} AUROC [95% CI]
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {report.models.map((model) => (
              <tr key={model.model_id} className="border-b last:border-0">
                <th scope="row" className="py-2 pr-4 font-mono text-xs font-normal">
                  {model.model_id}
                </th>
                <td className="py-2 pr-4 text-right">{model.n_rows}</td>
                <td className="py-2 pr-4 text-right">{model.n_hallucinations}</td>
                <td className="py-2 pr-4 text-right">{formatScore(model.accuracy)}</td>
                {detectorNames.map((name) => {
                  const d = model.detectors[name]
                  return (
                    <td key={name} className="whitespace-nowrap py-2 pr-4 text-right">
                      {d ? (
                        <>
                          {formatScore(d.auroc)}{' '}
                          <span className="text-muted-foreground">
                            [{formatScore(d.ci.low)}, {formatScore(d.ci.high)}]
                          </span>
                        </>
                      ) : (
                        <span className="text-muted-foreground">n/a</span>
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
