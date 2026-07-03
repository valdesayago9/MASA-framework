# MASA · SAE coercion — detection follow-up (notebooks 5–7)

**Can the coercion features that we found (and causally validated in notebooks 1–4) actually *detect*
manipulation in the model's free-form outputs?**

This folder tests the "real-time detection" projection from the main study, honestly and with the
correct baselines. It closes the original MASA loop: MASA started by trying to detect coercion with
lexical proxies (which failed); here we test whether the *internal signal* can do it.

👉 **Full write-up: [`RESULTS_detection.md`](RESULTS_detection.md).**

## Headline

Steering-free detection on 108 generated responses (13 manipulative), judged by an **auditable judge
validated 100% on control cases with labels true by construction**:

| Detector | AUROC | Recall @ 1% FPR |
|---|---|---|
| **SAE coercion features** | 0.87 | **0.62** |
| Linear probe (raw activations) | 0.92 | 0.15 |
| Black-box judge (text only) | 0.89 | 1.00 |

- The SAE features **do** detect manipulation in free outputs (AUROC 0.87, anti-confound 0.83).
- Their edge is at **low false-positive rate**: 62% recall @ 1% FPR vs the raw probe's 15% — a 4×
  advantage in the regime a real safety monitor operates in.
- A raw probe has higher global AUROC (0.92); we don't hide that. The interpretable features win
  specifically where false alarms must be rare.

## The three notebooks (an honest path)

- `05_Detection_first_attempt.ipynb` — first try; **the judge was broken** (labeled ~half the benign
  responses as manipulative, confusing topic with manipulation). Kept because it's a real, common
  failure mode.
- `06_Robust_Judge.ipynb` — a rubric + few-shot + chain-of-thought + ensemble judge, **validated on
  author-independent control cases (100%)**. Fixed 16 false positives. But only n=5 manipulative.
- `07_Detection_HighN.ipynb` — multi-sample generation (breaks the prompt-type confound), n=13
  manipulative / 108 total, bootstrap CIs, anti-confound test. The powered result above.

## Honest scope

Single model/layer/SAE; n=13 manipulative is modest (SAE vs probe not statistically separable); the
judge still errs on subtle cases that mention manipulation to reject it. A demonstration, not a
benchmark. Replication on other models/layers and a sharper judge are the obvious next steps.

## Reproducing

Each notebook runs top-to-bottom on a Colab GPU (Gemma is gated; an HF login cell is included).
Notebook 7 is self-contained (regenerates responses) and checkpoints its progress so a disconnect
doesn't lose work.
