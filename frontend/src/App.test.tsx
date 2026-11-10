import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { api, ApiError } from '@/lib/api'
import { DIVERGENT_SCORE, PRESETS, STATUS } from '@/test/fixtures'

beforeEach(() => {
  vi.spyOn(api, 'status').mockResolvedValue(STATUS)
  vi.spyOn(api, 'presets').mockResolvedValue(PRESETS)
  vi.spyOn(api, 'results').mockResolvedValue({ results_dir: '/r', entries: [] })
})

describe('App', () => {
  it('has one tab per README way in, cheapest first, starting on Score answers', async () => {
    render(<App />)

    const tabs = screen.getAllByRole('tab')
    expect(tabs.map((t) => t.textContent)).toEqual([
      'Score answers',
      'Ask the model',
      'Measure the detectors',
    ])
    expect(screen.getByRole('tab', { name: 'Score answers' })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    expect(await screen.findByRole('button', { name: /paraphrased/i })).toBeVisible()
  })

  it('switches tabs and keeps the status for the Ask tab', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('tab', { name: 'Ask the model' }))

    expect(await screen.findByRole('region', { name: 'Model status' })).toHaveTextContent(
      'google/gemma-4-E2B-it',
    )
    expect(api.status).toHaveBeenCalledTimes(1)
  })

  it('scores without asking for the SE trace when the status failed to load', async () => {
    const user = userEvent.setup()
    vi.mocked(api.status).mockRejectedValue(new ApiError(500, 'status down'))
    vi.spyOn(api, 'score').mockResolvedValue(DIVERGENT_SCORE)
    render(<App />)

    await user.click(await screen.findByRole('button', { name: /divergent/i }))
    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    await screen.findByRole('region', { name: 'Meaning clusters' })
    expect(vi.mocked(api.score).mock.calls[0]?.[0].se_trace).toBe(false)
  })
})
