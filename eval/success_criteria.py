"""Pass bars for the project's research success criteria.

The full table, with the command that measures each one, lives in
knowledge/usage_guides/QA_GUIDE.md ("What does success look like for the
detectors?").
"""

from typing import Final

# 1. Semantic Entropy separates hallucinated from correct TriviaQA answers.
SE_AUROC_TARGET: Final[float] = 0.75
# 2. Clustering by meaning beats token-level uncertainty by a clear margin.
SE_MINUS_PE_AUROC_TARGET: Final[float] = 0.05
# 3. Escalating to the judge raises end-to-end accuracy (percentage points).
ESCALATION_UPLIFT_PP_TARGET: Final[float] = 5.0
# 4. SE is higher on adversarial prompts than on correctly answered ones.
ADVERSARIAL_P_VALUE_TARGET: Final[float] = 0.01
