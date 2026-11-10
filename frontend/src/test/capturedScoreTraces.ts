/**
 * Real /api/score responses with `se_trace: true` for two presets, captured
 * from the playground backend running microsoft/deberta-large-mnli (the
 * pinned NLI_REVISION). Unlike the hand-built traces in fixtures.ts, these are
 * the exact comparisons the backend's greedy clustering made. Regenerate them
 * by POSTing each preset to /api/score with se_trace true if the presets or
 * the NLI model change.
 */

import type { ScoreResponse } from '@/lib/api'

export interface CapturedScore {
  question: string
  answers: string[]
  response: ScoreResponse
}

/** The paraphrased preset, scored with the replay recorded. */
export const PARAPHRASED_PRESET: CapturedScore = {
  "question": "What is the capital of France?",
  "answers": [
    "Paris",
    "The capital is Paris.",
    "It's Paris.",
    "Paris, France",
    "Paris is the capital of France.",
    "Paris",
    "The answer is Paris.",
    "France's capital is Paris.",
    "Paris.",
    "That would be Paris."
  ],
  "response": {
    "semantic_entropy": 0.0,
    "se_estimator": "likelihood_weighted",
    "predictive_entropy": 0.5999999999999999,
    "clusters": [
      {
        "member_indices": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7,
          8,
          9
        ],
        "representative_text": "Paris"
      }
    ],
    "n_answers": 10,
    "se_threshold": 1.0,
    "se_trace": {
      "clustering": {
        "comparisons": [
          {
            "answer_index": 1,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          },
          {
            "answer_index": 2,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          },
          {
            "answer_index": 3,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          },
          {
            "answer_index": 4,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          },
          {
            "answer_index": 5,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          },
          {
            "answer_index": 6,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          },
          {
            "answer_index": 7,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          },
          {
            "answer_index": 8,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          },
          {
            "answer_index": 9,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          }
        ],
        "placements": [
          {
            "answer_index": 0,
            "cluster_index": 0,
            "opened_new_cluster": true
          },
          {
            "answer_index": 1,
            "cluster_index": 0,
            "opened_new_cluster": false
          },
          {
            "answer_index": 2,
            "cluster_index": 0,
            "opened_new_cluster": false
          },
          {
            "answer_index": 3,
            "cluster_index": 0,
            "opened_new_cluster": false
          },
          {
            "answer_index": 4,
            "cluster_index": 0,
            "opened_new_cluster": false
          },
          {
            "answer_index": 5,
            "cluster_index": 0,
            "opened_new_cluster": false
          },
          {
            "answer_index": 6,
            "cluster_index": 0,
            "opened_new_cluster": false
          },
          {
            "answer_index": 7,
            "cluster_index": 0,
            "opened_new_cluster": false
          },
          {
            "answer_index": 8,
            "cluster_index": 0,
            "opened_new_cluster": false
          },
          {
            "answer_index": 9,
            "cluster_index": 0,
            "opened_new_cluster": false
          }
        ]
      },
      "estimator": "likelihood_weighted",
      "answer_likelihoods": [
        0.5488116360940264,
        0.5488116360940264,
        0.5488116360940264,
        0.5488116360940264,
        0.5488116360940264,
        0.5488116360940264,
        0.5488116360940264,
        0.5488116360940264,
        0.5488116360940264,
        0.5488116360940264
      ],
      "cluster_probabilities": [
        1.0
      ]
    }
  }
}

/** The divergent preset, scored with the replay recorded. */
export const DIVERGENT_PRESET: CapturedScore = {
  "question": "Who was the first woman to walk on Mars?",
  "answers": [
    "Sally Ride",
    "Valentina Tereshkova",
    "Peggy Whitson",
    "Sally Ride",
    "Christina Koch",
    "Mae Jemison",
    "No woman has walked on Mars.",
    "Valentina Tereshkova",
    "Eileen Collins",
    "Sally Ride"
  ],
  "response": {
    "semantic_entropy": 1.8343719702816235,
    "se_estimator": "likelihood_weighted",
    "predictive_entropy": 0.4,
    "clusters": [
      {
        "member_indices": [
          0,
          3,
          9
        ],
        "representative_text": "Sally Ride"
      },
      {
        "member_indices": [
          1,
          7
        ],
        "representative_text": "Valentina Tereshkova"
      },
      {
        "member_indices": [
          2
        ],
        "representative_text": "Peggy Whitson"
      },
      {
        "member_indices": [
          4
        ],
        "representative_text": "Christina Koch"
      },
      {
        "member_indices": [
          5
        ],
        "representative_text": "Mae Jemison"
      },
      {
        "member_indices": [
          6
        ],
        "representative_text": "No woman has walked on Mars."
      },
      {
        "member_indices": [
          8
        ],
        "representative_text": "Eileen Collins"
      }
    ],
    "n_answers": 10,
    "se_threshold": 1.0,
    "se_trace": {
      "clustering": {
        "comparisons": [
          {
            "answer_index": 1,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 2,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 2,
            "cluster_index": 1,
            "representative_index": 1,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 3,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          },
          {
            "answer_index": 4,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 4,
            "cluster_index": 1,
            "representative_index": 1,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 4,
            "cluster_index": 2,
            "representative_index": 2,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 5,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 5,
            "cluster_index": 1,
            "representative_index": 1,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 5,
            "cluster_index": 2,
            "representative_index": 2,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 5,
            "cluster_index": 3,
            "representative_index": 4,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 6,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 6,
            "cluster_index": 1,
            "representative_index": 1,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 6,
            "cluster_index": 2,
            "representative_index": 2,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 6,
            "cluster_index": 3,
            "representative_index": 4,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 6,
            "cluster_index": 4,
            "representative_index": 5,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 7,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 7,
            "cluster_index": 1,
            "representative_index": 1,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          },
          {
            "answer_index": 8,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 8,
            "cluster_index": 1,
            "representative_index": 1,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 8,
            "cluster_index": 2,
            "representative_index": 2,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 8,
            "cluster_index": 3,
            "representative_index": 4,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 8,
            "cluster_index": 4,
            "representative_index": 5,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 8,
            "cluster_index": 5,
            "representative_index": 6,
            "forward": "contradiction",
            "backward": null,
            "joined": false
          },
          {
            "answer_index": 9,
            "cluster_index": 0,
            "representative_index": 0,
            "forward": "entailment",
            "backward": "entailment",
            "joined": true
          }
        ],
        "placements": [
          {
            "answer_index": 0,
            "cluster_index": 0,
            "opened_new_cluster": true
          },
          {
            "answer_index": 1,
            "cluster_index": 1,
            "opened_new_cluster": true
          },
          {
            "answer_index": 2,
            "cluster_index": 2,
            "opened_new_cluster": true
          },
          {
            "answer_index": 3,
            "cluster_index": 0,
            "opened_new_cluster": false
          },
          {
            "answer_index": 4,
            "cluster_index": 3,
            "opened_new_cluster": true
          },
          {
            "answer_index": 5,
            "cluster_index": 4,
            "opened_new_cluster": true
          },
          {
            "answer_index": 6,
            "cluster_index": 5,
            "opened_new_cluster": true
          },
          {
            "answer_index": 7,
            "cluster_index": 1,
            "opened_new_cluster": false
          },
          {
            "answer_index": 8,
            "cluster_index": 6,
            "opened_new_cluster": true
          },
          {
            "answer_index": 9,
            "cluster_index": 0,
            "opened_new_cluster": false
          }
        ]
      },
      "estimator": "likelihood_weighted",
      "answer_likelihoods": [
        0.6703200460356393,
        0.6703200460356393,
        0.6703200460356393,
        0.6703200460356393,
        0.6703200460356393,
        0.6703200460356393,
        0.6703200460356392,
        0.6703200460356393,
        0.6703200460356393,
        0.6703200460356393
      ],
      "cluster_probabilities": [
        0.3,
        0.2,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1
      ]
    }
  }
}
