---
base_model: org/model
datasets:
- owner/sep-calibration-model
tags:
- semantic-entropy-probe
- hallucination-detection
---

# Semantic Entropy Probe for org/model

A linear classifier (logistic regression) that reads one hidden state of
[org/model](https://huggingface.co/org/model) and predicts whether the model's
semantic entropy (SE) on the question would be high. It replaces SE's N sampled answers
and their pairwise meaning checks with a single forward pass over the prompt.

Its score is **p(high SE)**: the probability that the discrete SE of 10 sampled
answers exceeds the split point γ* = 0.500 nats. It is **not** a hallucination
probability and **not** in nats.

Fitted with [hallucinations-math-lab](https://github.com/eladlaor/hallucinations-math-lab) at commit
[`eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee`](https://github.com/eladlaor/hallucinations-math-lab/tree/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee), from the calibration dataset
[owner/sep-calibration-model](https://huggingface.co/datasets/owner/sep-calibration-model).

## How do I use it?

With hallucinations-math-lab, enable the `semantic_entropy_probe` detector and point it here:

```
SEP_PROBE_PATH=hf://owner/sep-probe-model@v1
```

The detector downloads the probe, then refuses to score unless the model under test is
the same backbone at the same revision, with the same number of layers, hidden size,
chat template and answer instruction.

`SEP_THRESHOLD` is the escalation threshold on p(high SE); see
[What threshold should I use?](#what-threshold-should-i-use).

## Which backbone, prompt and layers does it read?

| | |
|---|---|
| Backbone | `org/model` at revision `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` (bfloat16) |
| Decoder layers (L) | 2 |
| Hidden size | 4 |
| Token position | TBG |
| Layers, this repo's numbering | 1, 2 |
| Layers, the paper's numbering (likely, not verified) | 2, 3 |
| Layer selection | fixed |
| Answer instruction | Answer the following question as briefly as possible. |
| Chat template SHA-256 | `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb` |

TBG ("token before generation") is the last token of the chat-formatted prompt, read
before any answer is generated. The selected layers' states are concatenated in
increasing order.

**Layer numbering.** In this repo, layer k is the hidden state after k decoder layers:
0 is the token embeddings and 2 is the last layer, which already has the
model's final norm applied. Kossen et al. appear to number the same states from 1, so
their layer k + 1 is probably our layer k (read from their code, not verified).

## How well does it detect hallucinations on held-out questions?

On 200 benchmark questions, none of them a training question (checked by
question id at publish time), of which 80 were hallucinated (the
greedy answer contains no ground-truth alias), each detector's AUROC against
hallucination, with a 95% bootstrap confidence interval:

| Detector | AUROC | 95% CI |
|---|---|---|
| `predictive_entropy` | 0.700 | 0.650–0.750 |
| `semantic_entropy_probe` | 0.740 | 0.690–0.790 |
| `semantic_entropy` | 0.800 | 0.750–0.850 |
| `discrete_semantic_entropy` | 0.780 | 0.730–0.830 |

AUROC 0.5 is chance and 1.0 a perfect ranking. The probe was trained to imitate binarized
SE, not to detect errors directly; read its AUROC beside SE's on the same questions.

## What did cross-validation measure on the training set?

Cross-validation (5 folds), with the layers fixed in advance, estimates the fitting procedure against binarized SE.
The last column is a probe on layer L alone, the pre-sweep baseline. Mean ± std over folds.

| Token position | Layers (ours) | Layers (paper, likely) | CV AUROC vs binarized SE | Layer-L AUROC vs binarized SE |
|---|---|---|---|---|
| TBG (served) | 1, 2 | 2, 3 | 0.770 ± 0.040 | 0.750 ± 0.050 |
| SLT | 1, 2 | 2, 3 | 0.690 ± 0.040 | 0.660 ± 0.050 |

Every single-layer AUROC, for both positions, is in `metrics.json` under `layer_sweep`.
The held-out report is under `heldout_benchmark`.

## What threshold should I use?

Training recommends **0.41234567891**: Youden's J on the out-of-fold
probabilities against model error. On the training rows it flags errors at a true-positive
rate of 0.700 and a false-positive rate of 0.300, at an escalation rate
of 0.450.

It is never applied implicitly: `SEP_THRESHOLD` is the threshold, and its default, 0.5, is
a placeholder that was not calibrated for escalation. Calibrate on your own development
set with `eval.calibrate_threshold`.

## How was it trained?

- **Data:** 10 calibration questions; 5 labelled high SE and 5 low SE.
- **Labels:** 10 answers sampled at temperature 1.0 (top-p 0.95,
  top-k 64, the checkpoint's own), at most 64 tokens each, clustered by
  meaning with `org/nli` at revision `cccccccccccccccccccccccccccccccccccccccc`. Discrete SE is split at
  γ* = 0.500 nats, the best split over midpoints between distinct SE values.
- **Classifier:** logistic regression, C = 1.0, l2 penalty, lbfgs solver,
  max_iter = 100 (converged in 12 iterations), feature scaling: none.
- **Software:** hidden states collected with the `hf_transformers` provider under transformers
  5.0.0; fitted with scikit-learn 1.8.0 on 2026-11-07T00:00:00+00:00.

## How does it differ from the paper?

It follows Kossen et al. (2024), with these deviations. Active for this probe:
**D1**.

| ID | Paper | Here |
|---|---|---|
| D1 | Plain-text few-shot "Answer:" prompt, no chat template | The model's chat template with a brief-answer instruction, zero-shot; TBG is therefore a different position |
| D2 | Llama-2, Mistral, Phi-3 | Gemma; the paper's layer choices do not transfer, so layers are swept |
| D3 | γ fitted on all rows before splitting | γ fitted per training fold; the final γ uses all rows |
| D4 | Layer window chosen on test-split AUROC | Chosen by inner cross-validation; the nested estimate is reported |
| D5 | Window ranked by AUROC against model error | Ranked by AUROC against binarized SE |
| D6 | About 2000 questions per dataset, several datasets | One calibration set |
| D7 | Most-likely answer sampled at T = 0.1 | Greedy answer, T = 0 |
| D8 | SE samples at T = 1.0, top-p 0.9, top-k 50 | The checkpoint's own top-p and top-k, so SE values are not directly comparable |
| D9 | No escalation threshold | A recommended threshold from Youden's J on out-of-fold rows |
| D10 | max_iter = 100 | 1000, only if 100 did not converge |
| D11 | Best split over a 100-point grid | Exhaustive search over midpoints between distinct SE values |
| D12 | No feature scaling | A standard scaler folded into the weights, only if max_iter = 1000 did not converge either |
| D13 | Layer 0 included | Layers 1..L only; at TBG, layer 0 is a constant vector |

## What are its limits?

- It works only with `org/model` at revision `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`, with the chat template and
  answer instruction above. Any other model, revision or prompt moves the hidden states;
  the detector refuses them rather than score.
- It imitates binarized discrete SE. Where SE misses a hallucination (a confidently
  repeated wrong answer), the probe misses it too.
- It was calibrated on the short-answer questions of one calibration set. Long answers,
  other domains and other languages are unmeasured.
- The vLLM provider can serve only layer L; a probe reading other layers needs the
  Hugging Face transformers provider. Agreement between the two providers' hidden states
  is unverified.
- It is not a correctness oracle: a low score does not mean the answer is right.

## How do I refit it without a GPU?

The dataset repo holds every calibration row, its sampled answers, meaning clusters, SE
values and the hidden states at every layer for both token positions. From a clone of
hallucinations-math-lab at the commit that fitted the probe:

```
git checkout eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
hf download owner/sep-calibration-model --repo-type dataset --revision v1 --local-dir sep-dataset
uv run python -m eval.train_sep_probe --dataset sep-dataset --output-dir sep-probe --layers 1-2
```

The trainer's options are the ones this probe was fitted with.

## Citation

```bibtex
@article{kossen2024semantic,
  title   = {Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs},
  author  = {Kossen, Jannik and Han, Jiatong and Razzak, Muhammed and Schut, Lisa and
             Malik, Shreshth and Gal, Yarin},
  journal = {arXiv preprint arXiv:2406.15927},
  year    = {2024}
}

@article{farquhar2024detecting,
  title   = {Detecting hallucinations in large language models using semantic entropy},
  author  = {Farquhar, Sebastian and Kossen, Jannik and Kuhn, Lorenz and Gal, Yarin},
  journal = {Nature},
  volume  = {630},
  pages   = {625--630},
  year    = {2024}
}
```
