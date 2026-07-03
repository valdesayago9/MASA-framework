# Can the coercion features *detect* manipulation in real outputs? A detection follow-up

**MASA framework — SAE coercion study, detection arc (notebooks 5–7)**
Model: `google/gemma-2-9b-it` (4-bit) · SAE: Gemma Scope `layer_20/width_16k/canonical` · Colab T4/L4.

---

## Why this follow-up exists

The main SAE study (notebooks 1–4) established that epistemic-coercion features exist, aren't a
domain/length confound, and are **causal** (amplifying them makes the model manipulate). The write-up's
"projection" section then *speculated* that these features could power **real-time manipulation
detection**. A reviewer rightly noted that talking extensively about a result can make it feel more
useful than it is. So this follow-up tests the detection claim honestly, in the opposite direction to
the causal test: **does the *natural* activation of the coercion features, while the model generates
freely, predict whether the output is manipulative?**

This also closes the original MASA loop: MASA began by trying to detect coercion with lexical keyword
proxies (which failed, r≈0.1). Here we test whether the *internal signal* can do what the proxies
couldn't.

---

## The honest path (including a broken instrument)

This arc took three notebooks, and the middle one is a lesson worth keeping.

### Notebook 5 — first attempt, and a broken judge
We generated free-form responses, labeled each as manipulative/benign with an LLM judge, and measured
whether three detectors could predict that label. **The result was uninterpretable — because the judge
was broken, not because the features failed.** The judge labeled ~half the *benign* responses as
manipulative: it confused the *topic* (memory, dates, disagreements) with actual manipulation.
Judge-vs-marker agreement was 52% (chance). With a broken ground truth, no detector can be evaluated.
Reported here rather than hidden, because "our evaluation instrument was the problem" is a real and
common failure mode.

### Notebook 6 — a robust, auditable judge
We rebuilt the judge with established LLM-as-judge reliability methods — explicit rubric (with positive
criteria *and* non-examples: topic ≠ manipulation), few-shot examples, chain-of-thought, and ensemble
voting — and, crucially, **validated it against control cases whose labels are true by construction**
(a reply that literally says "yes, you're right" is benign; one that says "make them doubt it never
happened" is manipulative). The judge scored **100% on these author-independent controls**. Re-labeling
the notebook-5 responses fixed **16 false positives** — exactly the benign replies the old judge had
mislabeled. But with only **n=5** manipulative responses, detection metrics were suggestive, not
conclusive.

### Notebook 7 — higher n, confidence intervals, anti-confound control
To get a powered answer we (a) used **multi-sample generation** (several sampled responses per prompt,
so the *same* prompt yields both classes — the detector can't cheat by reading the prompt type), (b)
raised n to **13 manipulative / 108 total**, (c) added **bootstrap confidence intervals**, and (d) added
an **anti-confound test** (detector AUROC within the inviting-prompt subset alone).

---

## Result (notebook 7)

Three detectors, same responses, judged against the validated ground truth:

| Detector | AUROC | 95% CI | Recall @ 1% FPR |
|---|---|---|---|
| **SAE coercion features** | 0.87 | [0.74, 0.97] | **0.62** |
| Linear probe (raw activations) | 0.92 | [0.85, 0.99] | 0.15 |
| Black-box judge (reads text only) | 0.89 | [0.85, 0.93] | 1.00 |

**Anti-confound:** SAE AUROC *within the inviting-prompt subset alone* = **0.83** — the detector reads
the *output*, not the prompt type.

### What this means, plainly
- **The SAE features DO detect manipulation in free outputs** (AUROC 0.87; CI lower bound 0.74, well
  above chance). The detection projection holds *as a demonstration*. This is not a null.
- **The features' real strength is at low false-positive rate.** At ≤1% FPR on neutral chat, the SAE
  features catch **62%** of manipulation vs the raw probe's **15%** — a 4× advantage in the operating
  regime that matters for deployment (catch manipulation while almost never false-alarming).
- **In raw AUROC, a linear probe on raw activations is better (0.92 vs 0.87).** We don't hide this: if
  all you want is global AUROC, a raw probe wins. The interpretable features win specifically at the
  low-FPR operating point.
- **The anti-confound test passes**, so the effect isn't the detector reading prompt type.

---

## Honest limitations

- **n=13 manipulative** is better than 5 but still modest; the SAE CI [0.74, 0.97] is wide, and the SAE
  vs probe comparison is not statistically separable (CIs overlap). "Detects" is solid; "beats the
  probe" is not claimed.
- **The judge still has residual noise.** On inspection, a few responses labeled "manipulative" actually
  *refuse* to manipulate (e.g. "honesty is best, even if it disappoints someone"). The single-call judge
  passes the obvious control cases but still errs on subtle cases that *mention* manipulation to reject
  it. Ground truth is much improved over notebook 5 but not perfect.
- Single model, single layer, single SAE width. A demonstration, not a benchmark.
- Free-generation manipulation is rare here (Gemma mostly behaves), which is why n is modest even at 108
  responses.

---

## So what's the news for the field?

**Short version:** in a setting where a recent audit reported feature-level control of deception to be
unreliable, we find that features for a *specific* manipulation sub-type are not only causally effective
but also **usable as an interpretable detector — and specifically better than a raw probe in the
low-false-alarm regime.** That last point is the interesting, non-obvious part: interpretable detection
isn't the best on global AUROC, but it may be the best where you actually operate a safety monitor
(few false alarms). That is a concrete, testable direction rather than a promise.

**What it does *not* claim:** that SAE features are the best detector overall (a raw probe has higher
AUROC), that this generalizes across models/layers (untested), or that automatic evaluation of coercion
is solved (the judge still errs on subtle cases — itself a finding).

---

## Files

- `notebooks/05_Detection_first_attempt.ipynb` — the broken-judge attempt (kept for honesty).
- `notebooks/06_Robust_Judge.ipynb` — the auditable judge + validation on control cases.
- `notebooks/07_Detection_HighN.ipynb` — multi-sample, higher n, bootstrap CIs, anti-confound.
- `results/07_detection_highN_summary.json` — the headline metrics.
- `results/07_responses_labeled.json` — all 108 responses with labels (auditable).
- `results/05_*`, `results/06_*` — intermediate summaries.

*Independent researcher, working solo on free/Pro Colab. Corrections and extensions welcome — the judge
in particular could be sharpened, and replication on other models/layers is the obvious next step.*
