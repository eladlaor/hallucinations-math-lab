# User Guide

## Table of Contents

- [Summary](#summary)
- [Asking a question](#asking-a-question)
- [Interpreting the output](#interpreting-the-output)

## Summary

`seh chat "<your question>"` runs the small model 10 times, decides whether its
answers agree in meaning, and either returns the small model's answer (when
confident) or escalates to Gemini (when not).

## Asking a question

```bash
seh chat "Who wrote the novel 1984?"
```

## Interpreting the output

The CLI prints three sections:

1. **Detection** — `SE` (Semantic Entropy in nats), `PE` (predictive entropy),
   whether the model `escalated`, and which model produced the `source` answer.
2. **Final Answer** — what the system actually decided to tell you.
3. **Judge Reasoning** (only when escalated) — the judge model's justification.

Rule of thumb: SE < 0.5 nats = strong agreement; SE > 1.5 nats = strong disagreement.
The threshold is configurable via `SE_THRESHOLD` in `.env`.
