# Session handoff: hallucinations-math-lab

## Table of Contents

- [Summary](#summary)
- [Open questions for the owner](#open-questions-for-the-owner)

## Summary

Read this first in every session. The sections below the summary are kept by the
escalate-to-owner skill: what the owner was asked, his answers, and the last handoff.

## Open questions for the owner

Kept by escalate-to-owner. A question stays here until the owner answers, so any later run can
pick it up with `escalate.py check`.

<!-- escalate-to-owner:inbox-cursor 1 -->
<!-- escalate-to-owner:open -->
<!-- q hallucinations-math-lab.0965052b kind=brief cursor=none -->
- **`hallucinations-math-lab.0965052b`**, asked 2026-11-10 11:07 IST
  - Was doing: SEP classifier branch done (c4442c4); GPU preflight done; awaiting cost approval for the first RunPod run.
  - Question: Approve the first GPU run for the SEP probe (about $0.20): May I rent a RunPod GPU for about 75 minutes to run the first SEP probe training and benchmark? Options: Cheap, with fallback / On-demand only / Not now. Recommended: Cheap, with fallback.
  - Plan per answer: On approval: gpu-training-expert runs collect, fit, held-out benchmark, compute_auroc on HEAD c4442c4; then private HF upload, fresh-clone check, B1, go public.
<!-- /q hallucinations-math-lab.0965052b -->
<!-- q hallucinations-math-lab.ab7d4501 kind=question cursor=1 -->
- **`hallucinations-math-lab.ab7d4501`**, asked 2026-11-10 13:23 IST
  - Was doing: B1 runbook final at ~/b1-hallucinations-math-lab; GPU validation run in progress
  - Question: hallucinations-math-lab: may I run the history cleanup (B1) next, and will you do two quick setup steps for it? Step 1, in the Claude session type: ! gh auth refresh -h github.com -s delete_repo (lets the cleanup be undone; one browser click). Step 2: the cleanup must run outside my locked worktree, so either type ! ~/b1-hallucinations-math-lab/B1_RUNBOOK.sh phase0 in the session yourself, or open a fresh Claude session in ~/code/hallucinations-math-lab and tell it to run that script phase by phase. What it does: renames today's private repo to an archive (kept, nothing deleted), creates a fresh private repo under the same name with the cleaned history (private folders, old cloud project id and AI co-author lines removed; files identical), and re-points local checkouts without touching your uncommitted README and glossary edits. Rehearsed twice on copies, both secret scanners clean. No cost. I recommend yes, right after the first GPU run finishes.
  - Plan per answer: After GPU validation: B1 phases 0-5 with output reviewed per phase; then release GPU run on the new sha, private HF upload, fresh-clone check, go public.
<!-- /q hallucinations-math-lab.ab7d4501 -->
<!-- /escalate-to-owner:open -->

### Answered

<!-- escalate-to-owner:answered -->
<!-- /escalate-to-owner:answered -->
