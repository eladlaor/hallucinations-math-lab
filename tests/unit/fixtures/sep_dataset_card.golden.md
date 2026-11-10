---
license: apache-2.0
pretty_name: SEP calibration data for google/gemma-4-E2B-it
language:
- en
task_categories:
- text-classification
tags:
- semantic-entropy-probe
- hallucination-detection
- hidden-states
source_datasets:
- mandarjoshi/trivia_qa
models:
- owner/sep-probe-model
---

# SEP calibration data for google/gemma-4-E2B-it

**License:** the `apache-2.0` license covers only the columns and files this repo derived.
The question text, ids and accepted answers come from TriviaQA and keep TriviaQA's
terms: check them before redistributing the questions. See
[the license section](#what-is-its-license-and-what-must-i-attribute).

The calibration data that fitted the semantic entropy probe
[owner/sep-probe-model](https://huggingface.co/owner/sep-probe-model) (tag `v1`). For each of 10 trivia
questions it holds 10 answers sampled from [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it),
their meaning clusters, the question's semantic entropy (SE), and the model's hidden
states at every layer, so the probe can be refitted or re-examined without a GPU.

The probe was fitted with [hallucinations-math-lab](https://github.com/eladlaor/hallucinations-math-lab) at commit
[`eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee`](https://github.com/eladlaor/hallucinations-math-lab/tree/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee); the commit that collected the data is not
recorded.

## What is in each file?

| File | Content |
|---|---|
| `rows.parquet` | One row per question; the columns are below. |
| `hidden_states.safetensors` | Two float32 tensors, `tbg` and `slt`, each [10, 3, 4]: the hidden state after 0..2 decoder layers, at each token position. The header's `question_ids` lists the question ids in row order. |
| `collection_config.json` | The backbone, prompt, sampling and meaning-check settings the rows were collected with. |
| `training_log.jsonl` | One line per question, written as collection ran: SE, cluster count, answer lengths, greedy correctness, elapsed time. |

`tbg` ("token before generation") is the last token of the chat-formatted prompt, read
before any answer is generated. `slt` is the second-last token, read by teacher-forcing
the greedy answer.

## What does each row hold?

| Column | Content | Source |
|---|---|---|
| `id` | The question's TriviaQA id. | TriviaQA |
| `question` | The question text. | TriviaQA |
| `ground_truth_aliases` | The accepted answers. | TriviaQA |
| `sampled_answers` | 10 answers sampled at temperature 1.0 (top-p 0.95, top-k 64), at most 64 tokens each. | this repo |
| `cluster_ids` | Each sampled answer's meaning cluster, in sample order. | this repo |
| `se_discrete` | Discrete SE over the clusters, in nats. | this repo |
| `se_likelihood_weighted` | Likelihood-weighted SE, in nats. | this repo |
| `n_clusters` | The number of meaning clusters. | this repo |
| `greedy_answer` | The greedy (temperature 0) answer. | this repo |
| `greedy_correct` | Whether the greedy answer matches an accepted answer. | this repo |

Answers are clustered by meaning with `org/nli` at revision `cccccccccccccccccccccccccccccccccccccccc`.

## How do I refit the probe from it?

From a clone of hallucinations-math-lab at the commit that fitted the probe:

```
git checkout eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
hf download owner/sep-calibration-model --repo-type dataset --revision v1 --local-dir sep-dataset
uv run python -m eval.train_sep_probe --dataset sep-dataset --output-dir sep-probe --layers 1-2
```

The trainer's options are the ones the published probe was fitted with.

## Where does it come from?

- **Backbone:** `google/gemma-4-E2B-it` at revision `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`, 2 decoder layers,
  hidden size 4.
- **Collection:** hidden states collected with the `hf_transformers` provider under
  transformers 5.0.0 and torch 2.14.1+cu126 (CUDA 12.6).
- **Questions:** the `rc.nocontext` configuration of TriviaQA's
  `validation` split, [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa).
- **Code:** the probe was fitted with [hallucinations-math-lab](https://github.com/eladlaor/hallucinations-math-lab) at commit
  `eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee`. The commit that collected the data is not recorded.

## What is its license, and what must I attribute?

The columns and files this repo derived (the sampled and greedy answers, cluster ids, SE
values and correctness labels, the hidden states, and the logs) are released under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The answers and hidden states come from `google/gemma-4-E2B-it`, which
Google releases under the Apache License 2.0
([its license page](https://ai.google.dev/gemma/docs/gemma_4_license)). The code is under the MIT
license ([LICENSE](https://github.com/eladlaor/hallucinations-math-lab/blob/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee/LICENSE) at the commit that fitted the probe).

The question text, its ids and the accepted answers come from TriviaQA (Joshi et al.,
2017; [mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa)), whose license the Hub lists
as **unknown**. They are not relicensed here: check TriviaQA's terms
before redistributing the questions.

```bibtex
@inproceedings{joshi2017triviaqa,
  title     = {TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading
               Comprehension},
  author    = {Joshi, Mandar and Choi, Eunsol and Weld, Daniel S. and Zettlemoyer, Luke},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational
               Linguistics (ACL)},
  year      = {2017}
}
```
