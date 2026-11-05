/**
 * Cluster colours: one fixed categorical order (validated for colour-vision
 * deficiency on adjacent pairs, light and dark), assigned by cluster number —
 * never cycled. Clusters past the palette share a neutral colour. Text always
 * stays in the normal ink; the colour sits in a swatch beside the cluster
 * number, so colour is never the only signal. Values live in index.css as
 * --series-N custom properties, stepped separately for dark mode.
 */

export const CATEGORICAL_SLOTS = 8
const OTHER_FILL = 'var(--series-other)'

/** CSS colour of the cluster at 0-based `index`. */
export function clusterColour(index: number): string {
  if (index < 0 || index >= CATEGORICAL_SLOTS) return OTHER_FILL
  return `var(--series-${index + 1})`
}
