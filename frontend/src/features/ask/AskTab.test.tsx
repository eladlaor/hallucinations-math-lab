import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AskTab } from '@/features/ask/AskTab'
import { api, ApiError } from '@/lib/api'
import { README_HARDWARE_URL } from '@/lib/constants'
import { ESCALATED_CHAT, STATUS, STATUS_NO_JUDGE } from '@/test/fixtures'

const QUESTION = 'Who was the first woman to walk on Mars?'

async function ask(buttonName: 'Detect' | 'Chat') {
  const user = userEvent.setup()
  await user.type(screen.getByLabelText('Question'), QUESTION)
  await user.click(screen.getByRole('button', { name: buttonName }))
}

describe('AskTab', () => {
  it('shows the provider, the model and the judge in a status strip', () => {
    render(<AskTab status={STATUS} statusError={null} />)

    const strip = screen.getByRole('region', { name: 'Model status' })
    expect(strip).toHaveTextContent('hf_transformers')
    expect(strip).toHaveTextContent('google/gemma-4-E2B-it')
    expect(strip).toHaveTextContent(/judge.*configured/i)
  })

  it('disables Chat with the reason when the judge is not configured', async () => {
    render(<AskTab status={STATUS_NO_JUDGE} statusError={null} />)
    await userEvent.setup().type(screen.getByLabelText('Question'), QUESTION)

    expect(screen.getByRole('button', { name: 'Chat' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Detect' })).toBeEnabled()
    expect(screen.getByText(/GCP_PROJECT_ID is not set/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /hardware/i })).toHaveAttribute(
      'href',
      README_HARDWARE_URL,
    )
  })

  it('disables both actions when the status could not be loaded', () => {
    render(<AskTab status={null} statusError="connection refused" />)

    expect(screen.getByRole('button', { name: 'Detect' })).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent('connection refused')
  })

  it('shows an in-progress state with elapsed time while the model generates', async () => {
    vi.spyOn(api, 'ask').mockReturnValue(new Promise(() => {}))
    render(<AskTab status={STATUS} statusError={null} />)

    await ask('Detect')

    const progress = screen.getByRole('status', { name: 'Generation progress' })
    expect(progress).toHaveTextContent(/generating/i)
    expect(progress).toHaveTextContent('0:00')
    expect(screen.getByRole('button', { name: 'Detect' })).toBeDisabled()
  })

  it('shows the answer, the escalation, the scores, the judge and the clusters', async () => {
    vi.spyOn(api, 'ask').mockResolvedValue(ESCALATED_CHAT)
    render(<AskTab status={STATUS} statusError={null} />)

    await ask('Chat')

    expect(api.ask).toHaveBeenCalledWith(QUESTION, 'chat')
    const answer = await screen.findByRole('region', { name: 'Answer' })
    expect(answer).toHaveTextContent('No woman has walked on Mars.')
    expect(answer).toHaveTextContent(/Gemini judge/)
    expect(answer).toHaveTextContent('Sally Ride') // the model's own greedy answer
    expect(screen.getByRole('status', { name: 'Escalation decision' })).toHaveTextContent(
      /escalated/i,
    )
    expect(screen.getByRole('group', { name: 'Semantic Entropy (SE)' })).toHaveTextContent('1.834')
    expect(screen.getByRole('region', { name: 'Judge reasoning' })).toHaveTextContent(
      'No human has landed on Mars.',
    )
    const clusters = screen.getByRole('region', { name: 'Meaning clusters' })
    expect(within(clusters).getAllByRole('listitem', { name: /^Cluster \d+/ })).toHaveLength(7)
  })

  it("shows the backend's error message verbatim", async () => {
    vi.spyOn(api, 'ask').mockRejectedValue(
      new ApiError(500, 'Could not build the model under test: CUDA out of memory'),
    )
    render(<AskTab status={STATUS} statusError={null} />)

    await ask('Detect')

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Could not build the model under test: CUDA out of memory',
    )
    expect(screen.getByRole('button', { name: 'Detect' })).toBeEnabled()
  })
})
