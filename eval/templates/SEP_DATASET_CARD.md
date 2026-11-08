---
license: $license
pretty_name: SEP calibration data for $model_id
language:
- en
task_categories:
- text-classification
tags:
- semantic-entropy-probe
- hallucination-detection
- hidden-states
source_datasets:
- $source_dataset
models:
- $model_repo
---

# SEP calibration data for $model_id

**License:** the `$license` license covers only the columns and files this repo derived.
The question text, ids and accepted answers come from TriviaQA and keep TriviaQA's
terms: check them before redistributing the questions. See
[the license section](#what-is-its-license-and-what-must-i-attribute).

The calibration data that fitted the semantic entropy probe
[$model_repo]($hub_url/$model_repo) (tag `$tag`). For each of $n_rows trivia
questions it holds $sample_n answers sampled from [$model_id]($hub_url/$model_id),
their meaning clusters, the question's semantic entropy (SE), and the model's hidden
states at every layer, so the probe can be refitted or re-examined without a GPU.

The probe was fitted with [hallucinations-math-lab]($repo_url) at commit
[`$repo_commit`]($repo_url/tree/$repo_commit); the commit that collected the data is not
recorded.

## What is in each file?

| File | Content |
|---|---|
| `$rows_file` | One row per question; the columns are below. |
| `$tensors_file` | Two float32 tensors, `$tbg` and `$slt`, each [$n_rows, $num_states, $hidden_size]: the hidden state after 0..$num_hidden_layers decoder layers, at each token position. The header's `$ids_metadata_key` lists the question ids in row order. |
| `$collection_config_file` | The backbone, prompt, sampling and meaning-check settings the rows were collected with. |
| `$training_log_file` | One line per question, written as collection ran: SE, cluster count, answer lengths, greedy correctness, elapsed time. |

`$tbg` ("token before generation") is the last token of the chat-formatted prompt, read
before any answer is generated. `$slt` is the second-last token, read by teacher-forcing
the greedy answer.

## What does each row hold?

| Column | Content | Source |
|---|---|---|
| `id` | The question's TriviaQA id. | TriviaQA |
| `question` | The question text. | TriviaQA |
| `ground_truth_aliases` | The accepted answers. | TriviaQA |
| `sampled_answers` | $sample_n answers sampled at temperature $temperature (top-p $top_p, top-k $top_k), at most $max_tokens tokens each. | this repo |
| `cluster_ids` | Each sampled answer's meaning cluster, in sample order. | this repo |
| `se_discrete` | Discrete SE over the clusters, in nats. | this repo |
| `se_likelihood_weighted` | Likelihood-weighted SE, in nats. | this repo |
| `n_clusters` | The number of meaning clusters. | this repo |
| `greedy_answer` | The greedy (temperature 0) answer. | this repo |
| `greedy_correct` | Whether the greedy answer matches an accepted answer. | this repo |

Answers are clustered by meaning with `$nli_model_id` at revision `$nli_revision`.

## How do I refit the probe from it?

From a clone of hallucinations-math-lab at the commit that fitted the probe:

```
git checkout $repo_commit
hf download $dataset_repo --repo-type dataset --revision $tag --local-dir sep-dataset
$refit_command
```

The trainer's options are the ones the published probe was fitted with.

## Where does it come from?

- **Backbone:** `$model_id` at revision `$revision`, $num_hidden_layers decoder layers,
  hidden size $hidden_size.
- **Collection:** hidden states collected with the `$provider` provider under
  transformers $transformers_version.
- **Questions:** the `$source_dataset_config` configuration of TriviaQA's
  `$source_dataset_split` split, [$source_dataset]($hub_url/datasets/$source_dataset).
- **Code:** the probe was fitted with [hallucinations-math-lab]($repo_url) at commit
  `$repo_commit`. The commit that collected the data is not recorded.

## What is its license, and what must I attribute?

The columns and files this repo derived (the sampled and greedy answers, cluster ids, SE
values and correctness labels, the hidden states, and the logs) are released under the
[$license_name]($license_url). The answers and hidden states come from `$model_id`, which
$backbone_licensor releases under the $backbone_license_name
([its license page]($backbone_license_url)). The code is under the $code_license_name
license ([LICENSE]($code_license_url) at the commit that fitted the probe).

The question text, its ids and the accepted answers come from TriviaQA (Joshi et al.,
2017; [$source_dataset]($hub_url/datasets/$source_dataset)), whose license the Hub lists
as **$source_dataset_hub_license**. They are not relicensed here: check TriviaQA's terms
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
