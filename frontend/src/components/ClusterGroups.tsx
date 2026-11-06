import type { Cluster } from '@/lib/api'
import { clusterColour } from '@/lib/clusterPalette'

/**
 * The answers grouped by meaning: one block per cluster, each answer a chip
 * carrying its cluster number and colour. A thin bar above shows each
 * cluster's share of the answers — the distribution SE is the entropy of.
 */
export function ClusterGroups({ answers, clusters }: { answers: string[]; clusters: Cluster[] }) {
  const total = answers.length
  return (
    <section aria-label="Meaning clusters" className="space-y-4">
      <div className="flex h-2 w-full gap-0.5" aria-hidden="true">
        {clusters.map((cluster, i) => (
          <div
            key={cluster.member_indices.join('-')}
            className="h-full first:rounded-l-full last:rounded-r-full"
            style={{
              flexGrow: cluster.member_indices.length,
              background: clusterColour(i),
            }}
            title={`Cluster ${i + 1}: ${cluster.member_indices.length} of ${total}`}
          />
        ))}
      </div>
      <ol className="space-y-3">
        {clusters.map((cluster, i) => {
          const number = i + 1
          const size = cluster.member_indices.length
          const name = `Cluster ${number}, ${size} of ${total} answers`
          return (
            <li key={cluster.member_indices.join('-')} aria-label={name} className="space-y-2">
              <p className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                <span
                  className="inline-block size-2.5 rounded-sm"
                  style={{ background: clusterColour(i) }}
                  aria-hidden="true"
                />
                Cluster {number}
                <span className="font-normal">
                  · {size} of {total} answers
                </span>
              </p>
              <ul className="flex flex-wrap gap-2">
                {cluster.member_indices.map((answerIndex) => (
                  <li
                    key={answerIndex}
                    className="inline-flex max-w-full items-center gap-2 rounded-full border bg-card py-1 pl-1 pr-3 text-sm"
                    style={{ borderColor: clusterColour(i) }}
                  >
                    <span
                      className="inline-flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-[11px] font-semibold tabular-nums"
                      style={{ boxShadow: `inset 0 0 0 2px ${clusterColour(i)}` }}
                      title={`Cluster ${number}`}
                    >
                      {number}
                    </span>
                    <span className="truncate">{answers[answerIndex]}</span>
                  </li>
                ))}
              </ul>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
