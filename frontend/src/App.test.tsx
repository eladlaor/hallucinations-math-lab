import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import App from '@/App'
import { api } from '@/lib/api'
import { PRESETS, STATUS } from '@/test/fixtures'

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
})
