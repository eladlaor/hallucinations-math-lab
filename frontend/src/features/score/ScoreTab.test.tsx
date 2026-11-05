import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ScoreTab } from '@/features/score/ScoreTab'
import { api, ApiError } from '@/lib/api'
import { DIVERGENT_SCORE, PRESETS, TYPED_SCORE } from '@/test/fixtures'

beforeEach(() => {
  vi.spyOn(api, 'presets').mockResolvedValue(PRESETS)
})

async function loadDivergentAndScore() {
  const user = userEvent.setup()
  vi.spyOn(api, 'score').mockResolvedValue(DIVERGENT_SCORE)
  render(<ScoreTab defaultThreshold={1.0} />)
  await user.click(await screen.findByRole('button', { name: /divergent/i }))
  await user.click(screen.getByRole('button', { name: 'Score answers' }))
  await screen.findByRole('region', { name: 'Meaning clusters' })
  return user
}

describe('ScoreTab', () => {
  it('loads a preset into the question and answer fields', async () => {
    const user = userEvent.setup()
    render(<ScoreTab defaultThreshold={1.0} />)

    await user.click(await screen.findByRole('button', { name: /divergent/i }))

    expect(screen.getByLabelText('Question')).toHaveValue('Who was the first woman to walk on Mars?')
    expect(screen.getAllByRole('textbox', { name: /^Answer \d+$/ })).toHaveLength(10)
    expect(screen.getByLabelText('Answer 7')).toHaveValue('No woman has walked on Mars.')
  })

  it('sends the preset logprobs and shows SE, PE and the clusters', async () => {
    await loadDivergentAndScore()

    const request = vi.mocked(api.score).mock.calls[0]?.[0]
    expect(request?.answers[0]).toEqual({
      text: 'Sally Ride',
      token_logprobs: [{ token: 'Sally Ride', logprob: -0.4 }],
    })
    expect(screen.getByRole('group', { name: 'Semantic Entropy (SE)' })).toHaveTextContent('1.834')
    expect(screen.getByRole('group', { name: 'Predictive Entropy (PE)' })).toHaveTextContent(
      '0.400',
    )
    expect(screen.getByRole('group', { name: 'Meaning clusters' })).toHaveTextContent('7')

    const clusters = screen.getByRole('region', { name: 'Meaning clusters' })
    const first = within(clusters).getByRole('listitem', { name: /Cluster 1, 3 of 10 answers/ })
    expect(within(first).getAllByText('Sally Ride')).toHaveLength(3)
    expect(within(first).getAllByText('1')).toHaveLength(3) // the cluster number on each chip
  })

  it('flips escalate with the threshold slider, without a network request', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const user = await loadDivergentAndScore()
    const decision = screen.getByRole('status', { name: 'Escalation decision' })
    expect(decision).toHaveTextContent(/escalate/i)
    expect(decision).not.toHaveTextContent(/don't escalate/i)

    const thumb = screen.getByRole('slider', { name: 'Escalation threshold' })
    thumb.focus()
    await user.keyboard('{End}')

    expect(thumb).toHaveAttribute('aria-valuenow', '3')
    expect(decision).toHaveTextContent(/don't escalate/i)
    expect(api.score).toHaveBeenCalledTimes(1)
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('scores typed answers and explains why PE needs token logprobs', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'score').mockResolvedValue(TYPED_SCORE)
    render(<ScoreTab defaultThreshold={1.0} />)

    await user.type(screen.getByLabelText('Question'), 'Capital of France?')
    await user.type(screen.getByLabelText('Answer 1'), 'Paris')
    await user.click(screen.getByRole('button', { name: 'Add answer' }))
    await user.type(screen.getByLabelText('Answer 2'), 'Lyon')
    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    const pe = await screen.findByRole('group', { name: 'Predictive Entropy (PE)' })
    expect(pe).toHaveTextContent(/needs token logprobs/i)
    expect(screen.getByRole('group', { name: 'Semantic Entropy (SE)' })).toHaveTextContent('0.693')
    expect(vi.mocked(api.score).mock.calls[0]?.[0]).toEqual({
      question: 'Capital of France?',
      answers: [{ text: 'Paris' }, { text: 'Lyon' }],
    })
  })

  it("drops a preset answer's logprobs once its text is edited", async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'score').mockResolvedValue(TYPED_SCORE)
    render(<ScoreTab defaultThreshold={1.0} />)
    await user.click(await screen.findByRole('button', { name: /paraphrased/i }))

    await user.type(screen.getByLabelText('Answer 2'), '!')
    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    const answers = vi.mocked(api.score).mock.calls[0]?.[0].answers
    expect(answers?.[0]?.token_logprobs).toHaveLength(1)
    expect(answers?.[1]).toEqual({ text: 'The capital is Paris.!' })
  })

  it('removes an answer', async () => {
    const user = userEvent.setup()
    render(<ScoreTab defaultThreshold={1.0} />)
    await user.click(await screen.findByRole('button', { name: /paraphrased/i }))

    await user.click(screen.getByRole('button', { name: 'Remove answer 1' }))

    expect(screen.getAllByRole('textbox', { name: /^Answer \d+$/ })).toHaveLength(1)
    expect(screen.getByLabelText('Answer 1')).toHaveValue('The capital is Paris.')
  })

  it('says the first request loads DeBERTa while it waits', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'score').mockReturnValue(new Promise(() => {}))
    render(<ScoreTab defaultThreshold={1.0} />)
    await user.click(await screen.findByRole('button', { name: /paraphrased/i }))

    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    expect(screen.getByRole('status', { name: 'Scoring progress' })).toHaveTextContent(/DeBERTa/)
    expect(screen.getByRole('button', { name: 'Score answers' })).toBeDisabled()
  })

  it("shows the backend's error verbatim", async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'score').mockRejectedValue(new ApiError(500, "Failed to load NLI model 'x'"))
    render(<ScoreTab defaultThreshold={1.0} />)
    await user.click(await screen.findByRole('button', { name: /paraphrased/i }))

    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    expect(await screen.findByRole('alert')).toHaveTextContent("Failed to load NLI model 'x'")
  })
})
