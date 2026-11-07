---
base_model: $model_id
datasets:
- $dataset_repo
tags:
- semantic-entropy-probe
- hallucination-detection
---

# Semantic Entropy Probe for $model_id

A linear classifier (logistic regression) that reads one hidden state of
[$model_id](https://huggingface.co/$model_id) and predicts whether the model's
semantic entropy (SE) on the question would be high. It replaces SE's N sampled answers
and their pairwise meaning checks with a single forward pass over the prompt.

Its score is **p(high SE)**: the probability that the discrete SE of $sample_n sampled
answers exceeds the split point γ* = $gamma_nats nats. It is **not** a hallucination
probability and **not** in nats.

Fitted with [hallucinations-math-lab]($repo_url) at commit
[`$repo_commit`]($repo_url/tree/$repo_commit), from the calibration dataset
[$dataset_repo](https://huggingface.co/datasets/$dataset_repo).

## How do I use it?

With hallucinations-math-lab, enable the `semantic_entropy_probe` detector and point it here:

```
SEP_PROBE_PATH=$probe_uri
```

The detector downloads the probe, then refuses to score unless the model under test is
the same backbone at the same revision, with the same number of layers, hidden size,
chat template and answer instruction.

`SEP_THRESHOLD` is the escalation threshold on p(high SE); see
[What threshold should I use?](#what-threshold-should-i-use).

## Which backbone, prompt and layers does it read?

| | |
|---|---|
| Backbone | `$model_id` at revision `$revision` ($dtype) |
| Decoder layers (L) | $num_hidden_layers |
| Hidden size | $hidden_size |
| Token position | $token_position |
| Layers, this repo's numbering | $layers |
| Layers, the paper's numbering (likely, not verified) | $paper_layers |
| Layer selection | $layer_selection |
| Answer instruction | $answer_instruction |
| Chat template SHA-256 | `$chat_template_sha256` |

TBG ("token before generation") is the last token of the chat-formatted prompt, read
before any answer is generated. The selected layers' states are concatenated in
increasing order.

**Layer numbering.** In this repo, layer k is the hidden state after k decoder layers:
0 is the token embeddings and $num_hidden_layers is the last layer, which already has the
model's final norm applied. Kossen et al. appear to number the same states from 1, so
their layer k + 1 is probably our layer k (read from their code, not verified).

## How well does it detect hallucinations on held-out questions?

On $heldout_n_rows benchmark questions, none of them a training question (checked by
question id at publish time), of which $heldout_n_hallucinations were hallucinated (the
greedy answer contains no ground-truth alias), each detector's AUROC against
hallucination, with a $ci_percent% bootstrap confidence interval:

| Detector | AUROC | $ci_percent% CI |
|---|---|---|
$heldout_table

AUROC 0.5 is chance and 1.0 a perfect ranking. The probe was trained to imitate binarized
SE, not to detect errors directly; read its AUROC beside SE's on the same questions.

## What did cross-validation measure on the training set?

$cv_estimate
The last column is a probe on layer L alone, the pre-sweep baseline. Mean ± std over folds.

| Token position | Layers (ours) | Layers (paper, likely) | $cv_column | Layer-L AUROC vs binarized SE |
|---|---|---|---|---|
$selection_table

Every single-layer AUROC, for both positions, is in `metrics.json` under `layer_sweep`.
The held-out report is under `heldout_benchmark`.

## What threshold should I use?

Training recommends **$recommended_threshold**: Youden's J on the out-of-fold
probabilities against model error. On the training rows it flags errors at a true-positive
rate of $threshold_tpr and a false-positive rate of $threshold_fpr, at an escalation rate
of $threshold_escalation_rate.

It is never applied implicitly: `SEP_THRESHOLD` is the threshold, and its default, 0.5, is
a placeholder that was not calibrated for escalation. Calibrate on your own development
set with `eval.calibrate_threshold`.

## How was it trained?

- **Data:** $n_rows calibration questions; $n_high labelled high SE and $n_low low SE.
- **Labels:** $sample_n answers sampled at temperature $temperature (top-p $top_p,
  top-k $top_k, the checkpoint's own), at most $max_tokens tokens each, clustered by
  meaning with `$nli_model_id` at revision `$nli_revision`. Discrete SE is split at
  γ* = $gamma_nats nats, the best split over midpoints between distinct SE values.
- **Classifier:** logistic regression, C = $C, $penalty penalty, $solver solver,
  max_iter = $max_iter (converged in $n_iter iterations), feature scaling: $standardization.
- **Software:** hidden states collected with the `$provider` provider under transformers
  $transformers_version; fitted with scikit-learn $sklearn_version on $created_at.

## How does it differ from the paper?

It follows Kossen et al. (2024), with these deviations. Active for this probe:
**$deviations**.

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

- It works only with `$model_id` at revision `$revision`, with the chat template and
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
git checkout $repo_commit
hf download $dataset_repo --repo-type dataset --revision $tag --local-dir sep-dataset
$refit_command
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
