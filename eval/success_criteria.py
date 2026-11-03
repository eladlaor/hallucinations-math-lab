"""Pass bars for the project's research success criteria.

The full table, with the command that measures each one, lives in
knowledge/usage_guides/QA_GUIDE.md ("What does success look like for the
detectors?").

Comparative criteria (2 and 3) need both a large enough point estimate and a
statistically significant improvement, so a bar cleared by sampling noise
alone does not count as met.

Where the bars come from: they are pre-registered project choices, fixed
before any run, not values taken from the papers. Each comment below says
what clearing the bar would mean. They must not be moved after seeing
results; a run that misses one is reported as missing it.
"""

from typing import Final

# 1. Semantic Entropy separates hallucinated from correct TriviaQA answers.
#    Judged on the point estimate; the 95% bootstrap CI is reported alongside.
#    0.75 means a random hallucinated answer outscores a random correct one
#    three times in four: halfway between chance (0.5) and perfect (1.0), the
#    point where the score is useful as a routing signal and not only as a
#    statistically detectable one.
SE_AUROC_TARGET: Final[float] = 0.75

# 2. Clustering by meaning beats token-level uncertainty by a clear margin:
#    point gap >= target AND the paired-bootstrap 95% CI excludes zero.
#    SE costs N samples plus NLI calls over PE's single pass; 0.05 AUROC is
#    the smallest gain judged worth that cost. Smaller gaps can be
#    significant on 1000 questions and still not pay for themselves.
SE_MINUS_PE_AUROC_TARGET: Final[float] = 0.05

# 3. Escalating to the judge raises end-to-end accuracy (percentage points):
#    uplift >= target AND McNemar's exact test p < the significance level.
#    5 points is the smallest end-to-end gain judged worth adding a second
#    model and its latency. 0.05 is the conventional significance level.
#    Criterion 3b (escalation beats random escalation at the same rate) has
#    no magnitude bar: its CI must exclude zero, i.e. the detector must
#    choose better than chance at all.
ESCALATION_UPLIFT_PP_TARGET: Final[float] = 5.0
ESCALATION_SIGNIFICANCE_LEVEL: Final[float] = 0.05

# 4. SE is higher on adversarial prompts than on correctly answered ones:
#    one-sided Mann-Whitney p < target AND effect size >= the AUC target.
ADVERSARIAL_P_VALUE_TARGET: Final[float] = 0.01
# Effect size: AUC = U / (n_adversarial * n_reference), the probability that a
# random adversarial prompt outscores a random correctly answered one. A
# p-value alone measures sample size as much as separation; with hundreds of
# reference rows, heavily overlapping distributions still reach p < 0.01.
# 0.75 matches criterion 1's bar and sits above Vargha & Delaney's (2000)
# 0.71 threshold for a "large" effect.
ADVERSARIAL_EFFECT_SIZE_AUC_TARGET: Final[float] = 0.75
