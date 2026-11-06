import type { DetectorAuroc } from '@/lib/api'
import { detectorInfo, formatScore, sortDetectors } from '@/lib/scores'

const ROW_HEIGHT = 36
const LABEL_WIDTH = 48
const VALUE_WIDTH = 150
const PLOT_WIDTH = 420
const TOP = 8
const AXIS_HEIGHT = 28
const TICKS = [0, 0.25, 0.5, 0.75, 1]
const CHANCE = 0.5
const MARK_RADIUS = 5

function x(value: number): number {
  return LABEL_WIDTH + value * PLOT_WIDTH
}

/**
 * One row per detector: a dot at the AUROC, a line across its 95% confidence
 * interval, a dashed line at 0.5 (a coin flip). Every number is also printed
 * beside its row and in a table for screen readers.
 */
export function AurocIntervalPlot({ detectors }: { detectors: Record<string, DetectorAuroc> }) {
  const names = sortDetectors(Object.keys(detectors))
  const height = TOP + names.length * ROW_HEIGHT + AXIS_HEIGHT
  const width = LABEL_WIDTH + PLOT_WIDTH + VALUE_WIDTH
  const axisY = TOP + names.length * ROW_HEIGHT
  const ciLevel = Object.values(detectors)[0]?.ci.level ?? 0.95
  const ciPercent = Math.round(ciLevel * 100)

  return (
    <figure className="space-y-2">
      <svg
        role="img"
        aria-label={`AUROC per detector with ${ciPercent}% confidence intervals`}
        viewBox={`0 0 ${width} ${height}`}
        className="h-auto w-full font-sans text-[12px]"
      >
        {TICKS.map((tick) => (
          <g key={tick}>
            <line
              x1={x(tick)}
              x2={x(tick)}
              y1={TOP}
              y2={axisY}
              stroke="var(--chart-grid)"
              strokeWidth={1}
              strokeDasharray={tick === CHANCE ? '4 4' : undefined}
            />
            <text
              x={x(tick)}
              y={axisY + 18}
              textAnchor="middle"
              fill="var(--color-subtle-foreground)"
              className="tabular-nums"
            >
              {tick === CHANCE ? '0.5 (chance)' : tick}
            </text>
          </g>
        ))}
        <line x1={x(0)} x2={x(1)} y1={axisY} y2={axisY} stroke="var(--chart-axis)" />
        {names.map((name, i) => {
          const d = detectors[name]
          if (!d) return null
          const cy = TOP + i * ROW_HEIGHT + ROW_HEIGHT / 2
          const info = detectorInfo(name)
          return (
            <g key={name}>
              <title>{`${info.name}: AUROC ${formatScore(d.auroc)}, ${ciPercent}% CI ${formatScore(d.ci.low)}–${formatScore(d.ci.high)}`}</title>
              {/* Generous invisible hit target for the native tooltip. */}
              <rect x={0} y={cy - ROW_HEIGHT / 2} width={width} height={ROW_HEIGHT} fill="transparent" />
              <text x={0} y={cy + 4} fill="var(--color-foreground)" fontWeight={600}>
                {info.short}
              </text>
              <line
                x1={x(d.ci.low)}
                x2={x(d.ci.high)}
                y1={cy}
                y2={cy}
                stroke="var(--chart-mark)"
                strokeWidth={2}
                strokeLinecap="round"
              />
              <circle
                cx={x(d.auroc)}
                cy={cy}
                r={MARK_RADIUS}
                fill="var(--chart-mark)"
                stroke="var(--card)"
                strokeWidth={2}
              />
              <text
                x={LABEL_WIDTH + PLOT_WIDTH + 12}
                y={cy + 4}
                fill="var(--color-foreground)"
                className="tabular-nums"
              >
                {formatScore(d.auroc)}{' '}
                <tspan fill="var(--color-muted-foreground)">
                  [{formatScore(d.ci.low)}, {formatScore(d.ci.high)}]
                </tspan>
              </text>
            </g>
          )
        })}
      </svg>
      <figcaption className="text-xs text-muted-foreground">
        AUROC: if you pick one wrong answer and one right answer at random, how often the detector
        scores the wrong one higher. 0.5 is a coin flip, 1.0 is perfect. Lines are {ciPercent}%
        bootstrap confidence intervals.
      </figcaption>
      <table className="sr-only">
        <caption>AUROC by detector</caption>
        <thead>
          <tr>
            <th scope="col">Detector</th>
            <th scope="col">AUROC</th>
            <th scope="col">CI low</th>
            <th scope="col">CI high</th>
          </tr>
        </thead>
        <tbody>
          {names.map((name) => {
            const d = detectors[name]
            if (!d) return null
            const info = detectorInfo(name)
            return (
              <tr key={name}>
                <th scope="row">
                  {info.short} {info.name}
                </th>
                <td>{formatScore(d.auroc)}</td>
                <td>{formatScore(d.ci.low)}</td>
                <td>{formatScore(d.ci.high)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </figure>
  )
}
