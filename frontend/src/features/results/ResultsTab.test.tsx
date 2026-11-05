import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { ResultsTab } from '@/features/results/ResultsTab'
import { api, ApiError } from '@/lib/api'
import { BENCHMARK_COMMAND } from '@/lib/constants'
import { RUN_REPORT, SWEEP_REPORT } from '@/test/fixtures'

const LISTING = {
  results_dir: '/repo/eval/results',
  entries: [
    { path: 'sweep', kind: 'sweep' as const, n_files: 2 },
    { path: 'triviaqa_200.jsonl', kind: 'run' as const, n_files: 1 },
  ],
}

describe('ResultsTab', () => {
  it('shows the command that produces results when there are none', async () => {
    vi.spyOn(api, 'results').mockResolvedValue({ results_dir: '/repo/eval/results', entries: [] })
    render(<ResultsTab />)

    expect(await screen.findByText(BENCHMARK_COMMAND)).toBeInTheDocument()
  })

  it('shows per-detector AUROC with its confidence interval for a run', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'results').mockResolvedValue(LISTING)
    vi.spyOn(api, 'resultsReport').mockResolvedValue({
      path: 'triviaqa_200.jsonl',
      kind: 'run',
      report: RUN_REPORT,
    })
    render(<ResultsTab />)

    await user.click(await screen.findByRole('button', { name: /triviaqa_200\.jsonl/ }))

    expect(api.resultsReport).toHaveBeenCalledWith('triviaqa_200.jsonl')
    const plot = await screen.findByRole('table', { name: /AUROC by detector/ })
    const seRow = within(plot).getByRole('row', { name: /SE/ })
    expect(seRow).toHaveTextContent('0.812')
    expect(seRow).toHaveTextContent('0.751')
    expect(seRow).toHaveTextContent('0.870')
    expect(screen.getByText(/200 questions scored/)).toBeInTheDocument()
    expect(screen.getByText(/87 hallucinated/)).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /AUROC/ })).toBeInTheDocument()
  })

  it('shows the model comparison table for a sweep', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'results').mockResolvedValue(LISTING)
    vi.spyOn(api, 'resultsReport').mockResolvedValue({
      path: 'sweep',
      kind: 'sweep',
      report: SWEEP_REPORT,
    })
    render(<ResultsTab />)

    await user.click(await screen.findByRole('button', { name: /sweep/ }))

    const table = await screen.findByRole('table', { name: /Model comparison/ })
    const rows = within(table).getAllByRole('row')
    expect(rows).toHaveLength(3) // header + two models
    expect(rows[2]).toHaveTextContent('google/gemma-4-E4B-it')
    expect(rows[2]).toHaveTextContent('0.695')
    expect(rows[2]).toHaveTextContent('n/a') // no PE in that run
  })

  it("shows the backend's error when a file cannot be scored", async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'results').mockResolvedValue(LISTING)
    vi.spyOn(api, 'resultsReport').mockRejectedValue(
      new ApiError(422, 'AUROC needs both correct and hallucinated rows'),
    )
    render(<ResultsTab />)

    await user.click(await screen.findByRole('button', { name: /triviaqa_200\.jsonl/ }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'AUROC needs both correct and hallucinated rows',
    )
  })
})
