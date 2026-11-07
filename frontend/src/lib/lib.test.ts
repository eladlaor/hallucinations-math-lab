import { afterEach, describe, expect, it, vi } from 'vitest'

import { api, ApiError } from '@/lib/api'
import { logger } from '@/lib/logger'
import { DetectorName } from '@/lib/constants'
import {
  detectorInfo,
  formatDetectorScore,
  formatElapsed,
  formatScore,
  shouldEscalate,
  thresholdSliderMax,
} from '@/lib/scores'

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function loggedEntries(spy: ReturnType<typeof vi.spyOn>): Record<string, unknown>[] {
  return spy.mock.calls.map((call: unknown[]) => JSON.parse(call[0] as string))
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('logger', () => {
  it('emits one JSON object with level, message and context', () => {
    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    logger.warn('something odd', { id: 7, error: new Error('boom') })

    const [entry] = loggedEntries(spy)
    expect(entry).toMatchObject({
      level: 'warn',
      message: 'something odd',
      id: 7,
      error: { name: 'Error', message: 'boom' },
    })
    expect(typeof entry?.ts).toBe('string')
  })
})

describe('api client', () => {
  it('logs start and success for a call', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse([])))
    const spy = vi.mocked(console.info)

    await api.presets()

    const messages = loggedEntries(spy).map((e) => [e.message, e.call])
    expect(messages).toEqual([
      ['api call started', 'presets'],
      ['api call succeeded', 'presets'],
    ])
  })

  it("throws the backend's message verbatim and logs the failure", async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(jsonResponse({ detail: 'model weights not found' }, 500)),
    )

    const error = await api.ask('Q?', 'detect').catch((e: unknown) => e)

    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).message).toBe('model weights not found')
    expect((error as ApiError).status).toBe(500)
    const failure = loggedEntries(vi.mocked(console.error)).find(
      (e) => e.message === 'api call failed',
    )
    expect(failure).toMatchObject({ call: 'ask', status: 500 })
  })

  it('turns validation errors into one readable line', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        jsonResponse({ detail: [{ loc: ['body', 'answers'], msg: 'too short' }] }, 422),
      ),
    )

    await expect(api.score({ question: 'Q?', answers: [] })).rejects.toThrow(
      'body.answers: too short',
    )
  })

  it('sends the report path as a query parameter', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}))
    vi.stubGlobal('fetch', fetchMock)

    await api.resultsReport('sweep dir/a.jsonl')

    expect(fetchMock.mock.calls[0]?.[0]).toBe('/api/results/report?path=sweep+dir%2Fa.jsonl')
  })
})

describe('scores', () => {
  it('formats to three decimals without a negative zero', () => {
    expect(formatScore(-1e-12)).toBe('0.000')
    expect(formatScore(1.8343719)).toBe('1.834')
    expect(formatScore(0.6)).toBe('0.600')
  })

  it('escalates only strictly above the threshold', () => {
    expect(shouldEscalate(1.834, 1.0)).toBe(true)
    expect(shouldEscalate(1.0, 1.0)).toBe(false)
  })

  it('lets the slider reach the largest possible SE', () => {
    expect(thresholdSliderMax(10)).toBe(3)
    expect(thresholdSliderMax(50)).toBeCloseTo(4.0)
  })

  it('shows SEP as a percentage and the entropies as plain numbers', () => {
    expect(formatDetectorScore(DetectorName.SEMANTIC_ENTROPY_PROBE, 0.7234)).toBe('72.3%')
    // numberDecimals applies only to the number scale; a probability keeps PERCENT_DECIMALS.
    expect(formatDetectorScore(DetectorName.SEMANTIC_ENTROPY_PROBE, 0.5, 2)).toBe('50.0%')
    expect(formatDetectorScore(DetectorName.SEMANTIC_ENTROPY, 1.8343719)).toBe('1.834')
    expect(formatDetectorScore(DetectorName.SEMANTIC_ENTROPY, 1.0, 2)).toBe('1.00')
    expect(formatDetectorScore('some_custom_detector', 0.25)).toBe('0.250')
  })

  it('explains SEP as P(high SE), not in nats', () => {
    const { meaning } = detectorInfo(DetectorName.SEMANTIC_ENTROPY_PROBE)
    expect(meaning).toContain('P(high SE)')
    expect(meaning).not.toContain('nats')
  })

  it('formats elapsed time as minutes and seconds', () => {
    expect(formatElapsed(65_400)).toBe('1:05')
  })
})
