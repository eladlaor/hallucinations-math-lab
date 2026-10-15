# Caveats: Theory vs. Production

> No-assumed-knowledge explanations of where Semantic Entropy / Predictive Entropy /
> SEP theory meets real-world production constraints. Written for talk-prep depth —
> each caveat explains the concept from scratch, then states the failure, then gives
> a one-sentence stage-ready version.

## Table of contents

- [1. NLI is doing unverified philosophical labor](#1-nli-is-doing-unverified-philosophical-labor)
- [2. The entailment threshold and clustering are order-dependent](#2-the-entailment-threshold-and-clustering-are-order-dependent)
- [3. N is small and entropy resolution is coarse](#3-n-is-small-and-entropy-resolution-is-coarse)
- [4. Cost is not "a bit more" — it's a different regime](#4-cost-is-not-a-bit-more--its-a-different-regime)
- [5. Domain shift breaks everything silently](#5-domain-shift-breaks-everything-silently)
- [6. SEP inherits SE's mistakes AND adds a generalization gap](#6-sep-inherits-ses-mistakes-and-adds-a-generalization-gap)
- [7. "Hallucination" itself is being operationalized narrowly](#7-hallucination-itself-is-being-operationalized-narrowly)

---

## 1. NLI is doing unverified philosophical labor

**What is NLI, first.** NLI = Natural Language Inference. It's a specific, narrow ML task: you give a model two sentences — a "premise" and a "hypothesis" — and it tells you if the hypothesis is *entailed* (must be true if the premise is true), *contradicted*, or *neutral* (unrelated / can't tell). Example: premise "A dog is running in the park," hypothesis "An animal is outside" → entailment. That's the whole task. Models are trained on datasets of exactly these triples (sentence, sentence, label).

**Why SE needs it.** To compute Semantic Entropy, you generate 10 different answers from the LLM, and you need to group them by "do these mean the same thing." There is no direct way to check "same meaning" in code — meaning isn't a number you can compute. So the SE method borrows the NLI model as a stand-in: "if these two answers entail each other in both directions, call them the same meaning."

**Why that's "unverified philosophical labor."** You're taking a genuinely hard, contested question ("what does it mean for two texts to have the same meaning?" — this is actually a real, unresolved debate in linguistics and philosophy of language) and quietly outsourcing the answer to a different neural network that was trained for a slightly different job (labeling isolated sentence pairs, not comparing free-form QA answers). Nobody proved this substitution is correct — it's a convenient approximation the field adopted because it's the best available tool, not because it's known to track "meaning" faithfully.

**The concrete failure this causes:** the NLI model can be wrong, biased, or just poorly calibrated on the specific kind of text you're feeding it — and your Semantic Entropy score inherits every one of those errors silently. You have no way to see it happening; you just get a number out, and you have to trust that the number reflects real semantic (dis)agreement.

**One sentence to say on stage:** "SE doesn't measure meaning directly — it measures what a separate NLI model *thinks* about meaning, and that model can be wrong in ways you can't see."

---

## 2. The entailment threshold and clustering are order-dependent

**What "clustering" means here.** You have 10 generated answers. You want to sort them into groups ("clusters") where everything in a group means the same thing. The ideal way to do this would be to compare *every* answer to *every other* answer (that's 10×10 = 100 comparisons, technically ~45 unique pairs) and build clusters from the full picture.

**What the code actually does, and why.** Comparing every pair against every other pair gets expensive fast as N grows (that's the "O(N²)" — grows with the square of the number of samples). So the code takes a shortcut: it processes answers one at a time, in whatever order they came out of the LLM, and for each new answer it only checks: "does this match the *first* answer that started each existing group?" If yes, join that group. If no, start a new group. This is much cheaper (that's the "O(N·k)" — grows with N times the number of groups, not N squared).

**Why that shortcut causes a real problem.** Which answer becomes each group's "representative" (the one new answers get compared against) depends entirely on *which order the answers happened to come out in*. If answer #3 had come out first instead of answer #1, it might have become the representative instead — and a different set of later answers might have matched against it. Same 10 answers, different arrival order, potentially different final grouping, potentially different entropy number.

**Why this matters:** you'd expect a measurement like "how much do these 10 answers disagree" to only depend on *what* the answers say, not on the arbitrary order the LLM happened to emit them in. But with this algorithm, order can matter. That's a hidden source of noise in the score that has nothing to do with the model's actual uncertainty.

**One sentence to say on stage:** "The clustering is a cheap approximation — it only checks new answers against the first answer in each group, so in principle, the exact same 10 answers in a different order could cluster slightly differently."

---

## 3. N is small and entropy resolution is coarse

**What "N" is.** N is the number of times you sample the LLM for the same question — in this repo, N=10.

**What "resolution" means here.** Semantic Entropy is a number computed from how the 10 answers split across clusters. With only 10 samples, there's a limited number of possible splits: all 10 in one cluster, 9-and-1, 8-and-2, etc. Each possible split produces one specific entropy value — there's no way to get a value "in between" two of these, because you can't have a fractional sample. So the set of possible SE scores you could ever see is a small, fixed list of numbers, not a smooth continuous range.

**Why that's a problem.** Imagine the *true* underlying uncertainty of the model is small but real — say, 1 time in 20 it would give a wrong, different answer. With only N=10 samples, you have a real chance of *never seeing that divergent answer at all* in your 10 draws (it only shows up 1 in 20 times on average) — so your 10 samples might all land in the same cluster, giving you an entropy of 0, meaning "totally confident," when the model actually does have a rare failure mode. To catch rarer disagreements reliably, you'd need more samples — but each additional sample means one more full LLM generation, which costs more time and money.

**One sentence to say on stage:** "With only 10 samples, rare hallucinations — the kind that show up 1 in 20 or 1 in 50 times — can simply be missed by chance, and getting better at catching them costs more generations, which costs more money and latency."

---

## 4. Cost is not "a bit more" — it's a different regime

**The comparison being made.** Predictive Entropy is "free" because it's computed from numbers (token probabilities) the model already produces as a side effect of generating one single answer — no extra work. Semantic Entropy requires generating **10 separate full answers** to the same question, plus running the NLI model multiple times to compare them.

**What "hot path" means.** In a live product (like a chatbot answering a user in real time), the "hot path" is the sequence of steps that happen while the user is actively waiting for a response. Anything in the hot path directly adds to how long the user stares at a loading spinner.

**Why this is a "different regime," not just "more expensive."** It's not like paying 10% more for a slightly better model. It's the difference between "generate 1 answer, done" and "generate 10 answers, cluster them, *then* give the user anything."

**But cost and latency are NOT the same axis, and conflating them is wrong.** The naive reading — 10 answers means 10× the wait — does not hold, because you do not loop 10 times. You issue **one** request with `n=10` (`SamplingParams` in vLLM, `num_return_sequences` in HF, `n` in the OpenAI API). Under that single request the prompt is prefilled **once**, its KV cache is shared across all N continuations, and those continuations decode as a batch. Decoding is memory-bandwidth-bound — the weights are read from HBM each step regardless — so widening the batch adds far less wall-clock than it adds compute.

So the honest split is roughly **10× the compute, but only ~1.1–1.5× the latency**, given batch headroom. Once sampling is batched, the latency bottleneck moves to the **NLI clustering**, which is inherently sequential (greedy clustering needs cluster *i*'s state before placing answer *i+1*) and in this repo runs one pair at a time.

> **Number-honesty:** the ~10× / ~1.1–1.5× figures are **reasoned from serving mechanics, not measured on this project's hardware.** `DetectorScore.latency_ms` records per-detector wall-clock; what is missing is a breakdown *within* SE (sampling vs. clustering). Until that benchmark exists, phrase these as "roughly" / "a fraction more" and never quote a hard multiplier as measured.

Two caveats that eat into the favorable latency number:

- Under a **saturated** server there is no batch headroom — your N sequences compete with other users' requests, and the 10× converts back into latency. The cost does not vanish; it moves to throughput.
- The request finishes when the **slowest** of the N finishes, not the average. Sampling variance means max-of-10 exceeds mean-of-1.

**Do NOT "fix" this by gating SE behind Predictive Entropy.** It is the obvious move and it is broken: PE's blind spot *is* the confident lie, so a PE gate never escalates precisely the cases SE exists to catch. See `src/hallucinations_math_lab/escalation.py`, whose module docstring makes this a hard rule (PE may be an `AnyAbove` OR-trigger, never an `AllAbove` gate). The sound cheap-first path is **SEP**, which approximates SE itself rather than substituting a different question.

**One sentence to say on stage:** "Semantic Entropy is roughly ten times the compute — but not ten times the wait, because the ten answers come back as one batch. Cost and latency are different axes."

---

## 5. Domain shift breaks everything silently

**What "domain" means here.** The type/genre of question and answer. Semantic Entropy was tested and validated on short factual questions with a single correct answer (like TriviaQA: "Who wrote 1984?" → "George Orwell"). That's one domain.

**What "domain shift" means.** Using the same method on a *different* kind of task than the one it was validated on — e.g., asking the model to write a summary, a poem, or a piece of code, instead of answering a short factual question.

**Why this breaks the whole premise, not just performance a little.** The entire method depends on being able to ask: "do these two answers mean the same thing?" For "Who wrote 1984?", that question has a clear answer — either both answers name Orwell or they don't. But for "summarize this article" or "write a function that sorts a list," there is no single correct meaning to compare against — two valid summaries can use completely different words, structure, and emphasis, and two valid pieces of code can be entirely different implementations, and both would be "correct" without being "the same." The NLI-based clustering has no principled way to decide if two different-but-both-valid summaries should be one cluster or ten — the concept of "semantic agreement" itself stops making sense for open-ended generation.

**Why "silently" matters:** the code will still run. It will still spit out a number. Nothing errors or warns you. It just quietly produces a meaningless entropy score dressed up as a real one, and if you don't already know the method's validated scope, you'd have no way to tell.

**One sentence to say on stage:** "This was proven on short factual QA — the moment you point it at open-ended generation like summaries or code, the underlying question 'do these mean the same thing' stops having a clean answer, and the method degrades quietly, with no error message telling you it broke."

---

## 6. SEP inherits SE's mistakes AND adds a generalization gap

**What SEP is, first.** Semantic Entropy Probe. Instead of generating 10 answers and clustering them every time (slow, expensive), you train a much simpler model once, offline: a single linear layer that looks at the LLM's internal state right after reading the *question* (before generating any answer), and tries to predict what the Semantic Entropy *would have been* if you'd done the full expensive process. Once trained, using it at inference time is just one fast matrix multiplication — cheap, like Predictive Entropy.

**Why it "inherits SE's mistakes."** To train this probe, you need training examples of (question, correct SE score) pairs — and the "correct SE score" for each training example was itself computed using the full, expensive, NLI-based Semantic Entropy process from caveat #1. So if that process is sometimes wrong or noisy (per caveats 1–3), the probe is trained to imitate those exact wrong/noisy numbers. It can't be more accurate than its own training labels — it's learning to mimic SE, flaws included, not learning some purer ground truth.

**What "generalization gap" means.** "Generalization" = does something that works on the data you tested it on also work on new, different data it hasn't seen. The probe is trained on the internal numerical representations ("hidden states") of one specific model — say, Gemma 2 9B. Those internal representations are specific to that model's architecture and training — a different model (even a same-family model of a different size) organizes its internal "thoughts" differently. So a probe trained on Gemma 2 9B's hidden states is essentially useless if you swap in a different model — you'd have to retrain it from scratch.

**One sentence to say on stage:** "SEP is a cheap shortcut that learns to imitate full Semantic Entropy — so it can't be more correct than SE's own flaws, and it's glued to the specific model it was trained on; swap models and you have to retrain it."

---

## 7. "Hallucination" itself is being operationalized narrowly

**What "operationalized" means.** Turning a fuzzy, informal concept ("did the model hallucinate?") into something a program can actually check with code — a concrete, checkable rule.

**What the actual rule in this repo is.** Look at `eval/compute_auroc.py`: an answer is scored as "correct" if it contains (as a substring, case-insensitive) one of a list of known-correct phrasings ("aliases") for the true answer. If it's *not* correct by that check, it's labeled a "hallucination." That's a very mechanical, narrow definition — it can't tell the difference between the model confidently making something up (a genuine hallucination) vs. giving a real answer that's just phrased in a way the alias list didn't anticipate vs. refusing to answer vs. a formatting mismatch.

**The deeper, sharper point — SE measures a different thing than "wrongness."** Semantic Entropy measures *how much the model's answers disagree with each other* across multiple tries. It does **not** measure whether those answers are actually *true*. These are two different failure modes:
- **"The model doesn't know, and it shows"** — ask it 10 times, you get 10 different guesses. High disagreement → high SE → correctly flagged as uncertain. This is called **epistemic uncertainty** ("epistemic" = relating to knowledge — the model lacks the knowledge, and that lack shows up as inconsistency).
- **"The model confidently learned something false"** — e.g., it always says a wrong birth year for someone, because that wrong fact is baked into its training data or weights. Ask it 10 times, you get the *same* wrong answer 10 times. Zero disagreement → SE = 0 → looks perfectly confident and trustworthy, even though it's completely wrong every time. This is sometimes called **aleatoric-like systematic error** — the wrongness is baked-in and consistent, not random.

**Why this is the sharpest caveat of all:** Semantic Entropy is fundamentally a *self-consistency* detector, not a *truth* detector. It only catches hallucinations that manifest as the model being inconsistent with itself. A model that is wrong the same way every single time will sail through SE completely undetected, scoring as if it were perfectly reliable.

**One sentence to say on stage:** "Semantic Entropy catches the model not knowing — when it's inconsistent across tries. It cannot catch the model confidently knowing something false — when it's wrong the same way every time — because that looks identical, mathematically, to being right every time."
