import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ScoreTab } from '@/features/score/ScoreTab'
import { api, ApiError } from '@/lib/api'
import {
  DIVERGENT_SCORE,
  DIVERGENT_SCORE_TRACED,
  PRESETS,
  STATUS_NO_TRACE,
  TYPED_SCORE,
} from '@/test/fixtures'

beforeEach(() => {
  vi.spyOn(api, 'presets').mockResolvedValue(PRESETS)
})

async function loadDivergentAndScore() {
  const user = userEvent.setup()
  vi.spyOn(api, 'score').mockResolvedValue(DIVERGENT_SCORE)
  render(<ScoreTab defaultThreshold={1.0} replayUnavailableReason={null} />)
  await user.click(await screen.findByRole('button', { name: /divergent/i }))
  await user.click(screen.getByRole('button', { name: 'Score answers' }))
  await screen.findByRole('region', { name: 'Meaning clusters' })
  return user
}

describe('ScoreTab', () => {
  it('loads a preset into the question and answer fields', async () => {
    const user = userEvent.setup()
    render(<ScoreTab defaultThreshold={1.0} replayUnavailableReason={null} />)

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
    const se = screen.getByRole('group', { name: 'Semantic Entropy (SE)' })
    expect(se).toHaveTextContent('1.834')
    expect(se).toHaveTextContent(/how likely the model found the answers/i)
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
    render(<ScoreTab defaultThreshold={1.0} replayUnavailableReason={null} />)

    await user.type(screen.getByLabelText('Question'), 'Capital of France?')
    await user.type(screen.getByLabelText('Answer 1'), 'Paris')
    await user.click(screen.getByRole('button', { name: 'Add answer' }))
    await user.type(screen.getByLabelText('Answer 2'), 'Lyon')
    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    const pe = await screen.findByRole('group', { name: 'Predictive Entropy (PE)' })
    expect(pe).toHaveTextContent(/needs token logprobs/i)
    const se = screen.getByRole('group', { name: 'Semantic Entropy (SE)' })
    expect(se).toHaveTextContent('0.693')
    expect(se).toHaveTextContent(/how many answers say it/i)
    expect(vi.mocked(api.score).mock.calls[0]?.[0]).toEqual({
      question: 'Capital of France?',
      answers: [{ text: 'Paris' }, { text: 'Lyon' }],
      se_trace: true,
    })
  })

  it('asks for the SE trace and replays the clustering one comparison at a time', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'score').mockResolvedValue(DIVERGENT_SCORE_TRACED)
    render(<ScoreTab defaultThreshold={1.0} replayUnavailableReason={null} />)
    await user.click(await screen.findByRole('button', { name: /divergent/i }))
    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    const replay = await screen.findByRole('region', { name: 'Clustering replay' })
    expect(vi.mocked(api.score).mock.calls[0]?.[0].se_trace).toBe(true)
    expect(within(replay).getByText(/^Step \d+ of \d+$/)).toHaveTextContent(
      'Step 1 of 33',
    )
    await user.click(within(replay).getByRole('button', { name: 'Next step' }))
    expect(within(replay).getByRole('region', { name: 'Current step' })).toHaveTextContent(
      'Who was the first woman to walk on Mars? Valentina Tereshkova',
    )
  })

  it('does not ask for the trace when the server cannot record it, and says why', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'score').mockResolvedValue(DIVERGENT_SCORE)
    render(
      <ScoreTab
        defaultThreshold={1.0}
        replayUnavailableReason={STATUS_NO_TRACE.se_trace_unavailable_reason}
      />,
    )
    await user.click(await screen.findByRole('button', { name: /divergent/i }))
    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    await screen.findByRole('region', { name: 'Meaning clusters' })
    expect(vi.mocked(api.score).mock.calls[0]?.[0].se_trace).toBe(false)
    expect(screen.getByText(/no step-by-step replay.*classify/i)).toBeInTheDocument()
  })

  it('does not ask for the trace while the server status is unknown, and says why', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'score').mockResolvedValue(DIVERGENT_SCORE)
    render(<ScoreTab defaultThreshold={1.0} replayUnavailableReason={undefined} />)
    await user.click(await screen.findByRole('button', { name: /divergent/i }))
    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    await screen.findByRole('region', { name: 'Meaning clusters' })
    expect(vi.mocked(api.score).mock.calls[0]?.[0].se_trace).toBe(false)
    expect(screen.getByText(/no step-by-step replay.*status has not loaded/i)).toBeInTheDocument()
  })

  it('shows no replay when the response carries no trace', async () => {
    await loadDivergentAndScore()

    expect(screen.queryByRole('region', { name: 'Clustering replay' })).not.toBeInTheDocument()
  })

  it("drops a preset answer's logprobs once its text is edited", async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'score').mockResolvedValue(TYPED_SCORE)
    render(<ScoreTab defaultThreshold={1.0} replayUnavailableReason={null} />)
    await user.click(await screen.findByRole('button', { name: /paraphrased/i }))

    await user.type(screen.getByLabelText('Answer 2'), '!')
    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    const answers = vi.mocked(api.score).mock.calls[0]?.[0].answers
    expect(answers?.[0]?.token_logprobs).toHaveLength(1)
    expect(answers?.[1]).toEqual({ text: 'The capital is Paris.!' })
  })

  it('removes an answer', async () => {
    const user = userEvent.setup()
    render(<ScoreTab defaultThreshold={1.0} replayUnavailableReason={null} />)
    await user.click(await screen.findByRole('button', { name: /paraphrased/i }))

    await user.click(screen.getByRole('button', { name: 'Remove answer 1' }))

    expect(screen.getAllByRole('textbox', { name: /^Answer \d+$/ })).toHaveLength(1)
    expect(screen.getByLabelText('Answer 1')).toHaveValue('The capital is Paris.')
  })

  it('says the first request loads DeBERTa while it waits', async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'score').mockReturnValue(new Promise(() => {}))
    render(<ScoreTab defaultThreshold={1.0} replayUnavailableReason={null} />)
    await user.click(await screen.findByRole('button', { name: /paraphrased/i }))

    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    expect(screen.getByRole('status', { name: 'Scoring progress' })).toHaveTextContent(/DeBERTa/)
    expect(screen.getByRole('button', { name: 'Score answers' })).toBeDisabled()
  })

  it("shows the backend's error verbatim", async () => {
    const user = userEvent.setup()
    vi.spyOn(api, 'score').mockRejectedValue(new ApiError(500, "Failed to load NLI model 'x'"))
    render(<ScoreTab defaultThreshold={1.0} replayUnavailableReason={null} />)
    await user.click(await screen.findByRole('button', { name: /paraphrased/i }))

    await user.click(screen.getByRole('button', { name: 'Score answers' }))

    expect(await screen.findByRole('alert')).toHaveTextContent("Failed to load NLI model 'x'")
  })
})
