import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ClusteringReplay } from '@/components/ClusteringReplay'
import { REPLAY_STEP_MS, ReplaySpeed } from '@/lib/constants'
import {
  DIVERGENT_QUESTION,
  DIVERGENT_SCORE,
  DIVERGENT_TEXTS,
  DIVERGENT_TRACE,
  TYPED_SCORE,
  TYPED_TRACE,
  WEIGHTED_QUESTION,
  WEIGHTED_SCORE_TRACED,
  WEIGHTED_TEXTS,
  WEIGHTED_TRACE,
} from '@/test/fixtures'
const N_STEPS = 1 + 25 + 7

function renderDivergent() {
  return render(
    <ClusteringReplay
      question={DIVERGENT_QUESTION}
      answers={DIVERGENT_TEXTS}
      clusters={DIVERGENT_SCORE.clusters}
      trace={DIVERGENT_TRACE}
      semanticEntropy={DIVERGENT_SCORE.semantic_entropy}
      typedAnswers={false}
    />,
  )
}

/** Discrete SE: from typed answers (the Score tab), or configured (the Ask tab). */
function renderDiscrete(typedAnswers: boolean) {
  return render(
    <ClusteringReplay
      question="Capital of France?"
      answers={['Paris', 'Lyon']}
      clusters={TYPED_SCORE.clusters}
      trace={TYPED_TRACE}
      semanticEntropy={TYPED_SCORE.semantic_entropy}
      typedAnswers={typedAnswers}
    />,
  )
}

const position = () => screen.getByText(/^Step \d+ of \d+$/)
const currentStep = () => screen.getByRole('region', { name: 'Current step' })

async function goToStep(user: ReturnType<typeof userEvent.setup>, step: number) {
  for (let i = 1; i < step; i += 1) {
    await user.click(screen.getByRole('button', { name: 'Next step' }))
  }
}

describe('ClusteringReplay, stepping by hand', () => {
  it('starts paused on the first answer opening meaning cluster 1', () => {
    renderDivergent()

    expect(position()).toHaveTextContent(`Step 1 of ${N_STEPS}`)
    expect(currentStep()).toHaveTextContent(/Answer 1.*Sally Ride/)
    expect(currentStep()).toHaveTextContent(/opens meaning cluster 1/i)
    expect(screen.getByRole('button', { name: 'Play' })).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Step back' })).toHaveAttribute('aria-disabled', 'true')
  })

  it('shows one comparison: the answer, the representative, both verdicts with the question in front', async () => {
    const user = userEvent.setup()
    renderDivergent()

    await goToStep(user, 2)

    const step = currentStep()
    expect(position()).toHaveTextContent(`Step 2 of ${N_STEPS}`)
    expect(step).toHaveTextContent(/Answer 2.*Valentina Tereshkova/)
    expect(step).toHaveTextContent(/meaning cluster 1.*representative.*answer 1/i)
    const forward = within(step).getByRole('group', { name: 'Forward verdict' })
    expect(forward).toHaveTextContent(`${DIVERGENT_QUESTION} Sally Ride`)
    expect(forward).toHaveTextContent(`${DIVERGENT_QUESTION} Valentina Tereshkova`)
    expect(forward).toHaveTextContent('contradiction')
    expect(within(step).getByRole('group', { name: 'Backward verdict' })).toHaveTextContent(
      /not run/i,
    )
    expect(step).toHaveTextContent(/opens meaning cluster 2/i)
  })

  it('shows a one-way entailment as not enough to join', async () => {
    const user = userEvent.setup()
    renderDivergent()
    // Answer 9 ("Eileen Collins") against meaning cluster 1 is the first of its comparisons:
    // 1 + (1+2+1+3+4+5+2) = 19 steps before it.
    await goToStep(user, 20)

    const step = currentStep()
    expect(step).toHaveTextContent(/Answer 9.*Eileen Collins/)
    expect(within(step).getByRole('group', { name: 'Forward verdict' })).toHaveTextContent(
      'entailment',
    )
    expect(within(step).getByRole('group', { name: 'Backward verdict' })).toHaveTextContent(
      'neutral',
    )
    expect(step).toHaveTextContent(/no match.*meaning cluster 2/i)
  })

  it('shows a join, and the meaning clusters so far', async () => {
    const user = userEvent.setup()
    renderDivergent()
    // Answer 4 ("Sally Ride") joins meaning cluster 1 on its first comparison: step 5.
    await goToStep(user, 5)

    expect(currentStep()).toHaveTextContent(/joins meaning cluster 1/i)
    const soFar = screen.getByRole('region', { name: 'Meaning clusters so far' })
    expect(within(soFar).getByRole('listitem', { name: /Meaning cluster 1: 2 answers/ })).toBeVisible()
  })

  it('steps back', async () => {
    const user = userEvent.setup()
    renderDivergent()
    await goToStep(user, 3)

    await user.click(screen.getByRole('button', { name: 'Step back' }))

    expect(position()).toHaveTextContent(`Step 2 of ${N_STEPS}`)
  })

  it('keeps keyboard focus on the step buttons at both ends of the replay', async () => {
    const user = userEvent.setup()
    renderDivergent()
    const next = screen.getByRole('button', { name: 'Next step' })
    const back = screen.getByRole('button', { name: 'Step back' })
    next.focus()

    for (let i = 1; i < N_STEPS; i += 1) await user.keyboard('{Enter}')
    expect(position()).toHaveTextContent(`Step ${N_STEPS} of ${N_STEPS}`)
    expect(next).toHaveFocus()
    await user.keyboard('{Enter}') // a no-op at the end
    expect(position()).toHaveTextContent(`Step ${N_STEPS} of ${N_STEPS}`)

    back.focus()
    for (let i = 1; i < N_STEPS; i += 1) await user.keyboard('{Enter}')
    expect(position()).toHaveTextContent(`Step 1 of ${N_STEPS}`)
    expect(back).toHaveFocus()
    await user.keyboard('{Enter}')
    expect(position()).toHaveTextContent(`Step 1 of ${N_STEPS}`)
  })

  it('never calls the meaning clusters buckets', () => {
    const { container } = renderDivergent()

    expect(container).toHaveTextContent(/meaning cluster/i)
    expect(container).not.toHaveTextContent(/bucket/i)
  })
})

describe('ClusteringReplay, the SE step', () => {
  it('ends with -sum p ln p filled in from likelihood weights, giving the reported SE', async () => {
    const user = userEvent.setup()
    renderDivergent()

    await goToStep(user, N_STEPS)

    expect(position()).toHaveTextContent(`Step ${N_STEPS} of ${N_STEPS}`)
    const entropy = screen.getByRole('region', { name: 'Semantic entropy' })
    expect(entropy).toHaveTextContent(/−Σ p\(c\) ln p\(c\)/)
    expect(entropy).toHaveTextContent(/likelihood/i)
    expect(entropy).toHaveTextContent('0.670') // exp(-0.4), each answer's weight
    const terms = within(entropy).getAllByRole('row', { name: /^Meaning cluster \d+$/ })
    expect(terms).toHaveLength(7)
    expect(terms[0]).toHaveTextContent('0.300')
    expect(terms[0]).toHaveTextContent('0.361')
    expect(entropy).toHaveTextContent('SE = 1.834 nats')
    expect(screen.getByRole('button', { name: 'Next step' })).toHaveAttribute('aria-disabled', 'true')
  })

  it("shows each meaning cluster's weight share as its p(c), not its size share", async () => {
    const user = userEvent.setup()
    render(
      <ClusteringReplay
        question={WEIGHTED_QUESTION}
        answers={WEIGHTED_TEXTS}
        clusters={WEIGHTED_SCORE_TRACED.clusters}
        trace={WEIGHTED_TRACE}
        semanticEntropy={WEIGHTED_SCORE_TRACED.semantic_entropy}
        typedAnswers={false}
      />,
    )

    // Answer 1 opens, answer 2 joins (1 comparison), answer 3 opens (1), 2 entropy terms.
    await goToStep(user, 1 + 2 + 2)

    const entropy = screen.getByRole('region', { name: 'Semantic entropy' })
    expect(entropy).toHaveTextContent('Weight / total')
    const [canberra, sydney] = within(entropy).getAllByRole('row', {
      name: /^Meaning cluster \d+$/,
    })
    expect(within(canberra as HTMLElement).getAllByRole('cell').map((c) => c.textContent)).toEqual(
      ['1', '0.400 / 1.000', '0.400', '0.367'],
    )
    expect(within(sydney as HTMLElement).getAllByRole('cell').map((c) => c.textContent)).toEqual(
      ['2', '0.600 / 1.000', '0.600', '0.306'],
    )
    expect(entropy).toHaveTextContent('SE = 0.673 nats')
  })

  it('fills the terms in one meaning cluster at a time', async () => {
    const user = userEvent.setup()
    renderDivergent()

    await goToStep(user, 1 + 25 + 1)

    const entropy = screen.getByRole('region', { name: 'Semantic entropy' })
    expect(within(entropy).getAllByRole('row', { name: /^Meaning cluster \d+$/ })).toHaveLength(1)
    expect(entropy).toHaveTextContent('so far 0.361')
  })

  it('shows p(c) = |c| / N for typed answers (discrete SE)', async () => {
    const user = userEvent.setup()
    renderDiscrete(true)

    await goToStep(user, 4)

    const entropy = screen.getByRole('region', { name: 'Semantic entropy' })
    expect(entropy).toHaveTextContent(/\|c\| \/ N/)
    expect(entropy).toHaveTextContent('1 / 2')
    expect(entropy).toHaveTextContent(/typed or edited answers/i)
    expect(entropy).not.toHaveTextContent(/likelihood/i)
    expect(entropy).toHaveTextContent('SE = 0.693 nats')
  })

  it('does not blame typed answers when discrete SE was configured (the Ask tab)', async () => {
    const user = userEvent.setup()
    renderDiscrete(false)

    await goToStep(user, 4)

    const entropy = screen.getByRole('region', { name: 'Semantic entropy' })
    expect(entropy).toHaveTextContent(/the discrete estimator/i)
    expect(entropy).not.toHaveTextContent(/typed/i)
  })
})

describe('ClusteringReplay, playback', () => {
  beforeEach(() => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  // fireEvent, not userEvent: its own pauses would wait on the fake clock.
  const click = (name: string) => fireEvent.click(screen.getByRole('button', { name }))
  const chooseSpeed = (speed: ReplaySpeed) =>
    fireEvent.change(screen.getByLabelText('Speed'), { target: { value: speed } })

  /** Advance the clock by `ms`, `times` times, letting React schedule the next step between. */
  function tick(ms: number, times = 1) {
    for (let i = 0; i < times; i += 1) {
      act(() => {
        vi.advanceTimersByTime(ms)
      })
    }
  }

  it('plays one step per tick at the chosen speed, and pauses', () => {
    renderDivergent()
    const slow = REPLAY_STEP_MS[ReplaySpeed.SLOW]

    click('Play')
    tick(slow)
    expect(position()).toHaveTextContent(`Step 2 of ${N_STEPS}`)
    tick(slow)
    expect(position()).toHaveTextContent(`Step 3 of ${N_STEPS}`)

    click('Pause')
    tick(slow, 3)
    expect(position()).toHaveTextContent(`Step 3 of ${N_STEPS}`)
  })

  it('changes the speed, from slow motion to real time', () => {
    renderDivergent()
    chooseSpeed(ReplaySpeed.SLOW_MOTION)

    click('Play')
    tick(REPLAY_STEP_MS[ReplaySpeed.SLOW_MOTION] - 1)
    expect(position()).toHaveTextContent(`Step 1 of ${N_STEPS}`)
    tick(1)
    expect(position()).toHaveTextContent(`Step 2 of ${N_STEPS}`)

    chooseSpeed(ReplaySpeed.REAL_TIME)
    tick(REPLAY_STEP_MS[ReplaySpeed.REAL_TIME], 3)
    expect(position()).toHaveTextContent(`Step 5 of ${N_STEPS}`)
  })

  it('stops on the SE step at the end', () => {
    renderDivergent()
    chooseSpeed(ReplaySpeed.REAL_TIME)

    click('Play')
    tick(REPLAY_STEP_MS[ReplaySpeed.REAL_TIME], N_STEPS + 5)

    expect(position()).toHaveTextContent(`Step ${N_STEPS} of ${N_STEPS}`)
    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Semantic entropy' })).toHaveTextContent(
      'SE = 1.834 nats',
    )
  })

  it('stays paused when stepping back after playing to the end', () => {
    renderDivergent()
    const realTime = REPLAY_STEP_MS[ReplaySpeed.REAL_TIME]
    chooseSpeed(ReplaySpeed.REAL_TIME)
    click('Play')
    tick(realTime, N_STEPS + 5)

    click('Step back')
    tick(realTime, 3)

    expect(position()).toHaveTextContent(`Step ${N_STEPS - 1} of ${N_STEPS}`)
    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument()
  })

  it('pauses on a manual step during playback', () => {
    renderDivergent()
    const slow = REPLAY_STEP_MS[ReplaySpeed.SLOW]
    click('Play')
    tick(slow)

    click('Next step')
    tick(slow, 3)

    expect(position()).toHaveTextContent(`Step 3 of ${N_STEPS}`)
    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument()
  })

  it('announces the current step to screen readers only while paused', () => {
    renderDivergent()
    expect(currentStep()).toHaveAttribute('aria-live', 'polite')
    expect(position()).not.toHaveAttribute('aria-live')
    expect(screen.queryByRole('status')).not.toBeInTheDocument()

    click('Play')
    expect(currentStep()).toHaveAttribute('aria-live', 'off')

    click('Pause')
    expect(currentStep()).toHaveAttribute('aria-live', 'polite')
  })

  it('plays again from the start once it has ended', () => {
    renderDivergent()
    for (let i = 1; i < N_STEPS; i += 1) click('Next step')

    click('Play')

    expect(position()).toHaveTextContent(`Step 1 of ${N_STEPS}`)
    expect(screen.getByRole('button', { name: 'Pause' })).toBeInTheDocument()
  })
})
