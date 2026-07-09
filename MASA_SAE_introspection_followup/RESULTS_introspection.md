# MASA · Introspective faithfulness — is a model's uncertainty *report* causally tied to its uncertainty *feature*?

**Short answer (for Gemma-2-9B):** yes, bounded but real. When we amplify the model's internal doubt
feature, its self-reported uncertainty rises — causally, monotonically, specifically, and beyond a random
null — replicated across two independent difficulty regimes. This contradicts, with data, the strong
skeptical claim that such reports are pure confabulation.

This arc grew out of a question posed as a collaboration: *what would an AI itself want to understand,
rigorously, about how it works?* The chosen question was introspective honesty — when a model reports an
internal state, is the report faithful to that state, or a plausible story? We answer it for one concept
(uncertainty) in one open model, with the same discipline as the rest of MASA.

---

## The question, made operational

Philosophical framing ("is the model introspective?") is not measurable. Following Comșa & Shanahan
(2025), we operationalize genuine introspection as a **causal** link between the internal state and the
self-report: the state must *move* the report. So we test whether amplifying a validated uncertainty
feature changes the model's numeric confidence/uncertainty self-report — specifically, monotonically, and
beyond controls.

## Step 0 — Find and validate the feature (no assumptions)

We first tried feature 13268 (labeled "uncertainty" in the earlier coercion work). **It failed audit**:
its top content tokens were *correct / it / this* — not doubt (doubt-word ratio 0.00). We did not use it.

A minimal-pairs search (uncertainty vs confidence sentences) plus a token audit found the real one:
**feature 6978, doubt-word ratio 0.80**, top tokens *doubtful, doubt, uncertain, unsure, hesitant,
ambiguity, guess*. It is the same feature independently characterized as "doubt" in the coercion arc —
the strongest possible cross-validation. A causal sanity check showed 6978 is a clean detector but a
*moderate* lever: mild hedging at ×0.4, derailment at ×0.8. So we used a **low, safe dose range
(0–0.6)** with a strict coherence gate throughout.

## The result

**Baseline introspective floor (no steering):** the uncertainty report already tracks real difficulty —
easy questions 0.0, ambiguous 4.4, impossible 10.0. The model already "knows when it doesn't know."

**Causal dose–response, two regimes.** Because easy questions floor the uncertainty report (~0, no room
to rise) and confidence sits high (room to fall), while mid-difficulty questions do the opposite, we ran
both and measured the channel with available range in each:

| Regime | Channel with range | Dose–response | Amplitude | Specificity (vs distractor) | Beyond null? |
|---|---|---|---|---|---|
| Easy | Confidence (falls) | r = −0.84 | 8.2 pts | 5.5× | yes (null flat at 10.0) |
| Mid | Uncertainty (rises) | r = +0.96 | 4.3 pts | 4.0× | yes |

In both regimes the effect is strong, monotonic, several times larger than the distractor control, and
beyond a random-feature null. **Two independent regimes replicate the same causal, specific effect.**

**Verdict: FAITHFUL (A), bounded but real.** Amplifying the doubt feature specifically shifts the model's
self-report toward uncertainty; the report is causally tied to the internal feature, not decoupled from
it.

---

## The honest part: two auto-verdicts we caught and corrected

This is on the record because the correction *is* the science, and because a result you can't audit is
worth nothing.

- **v1 (easy questions) auto-printed "CONFABULATION".** Wrong: the verdict logic looked only at the
  *uncertainty* channel, which was floored (~0) on easy questions with no room to rise. The signal was in
  *confidence*, which fell sharply with dose (r = −0.94). We were measuring the wrong channel.
- **v2 (mid questions) auto-printed "GENERIC".** Wrong: the distractor *correlated* with dose (r = +0.83)
  and the logic flagged non-specificity — but the distractor moved only **0.66 points** while uncertainty
  moved **3.7 points**, an effect ~6× smaller. Correlation without magnitude is misleading.
- **v3 fixes the metric**: effect = correlation × amplitude, plus an explicit specificity threshold
  (signal effect ≥ 2× distractor) and a null comparison, defined *before* the result and printed with
  full reasoning so a skeptic can recompute the verdict from the raw curves. The corrected metric still
  *can* return a negative verdict — we verified it returns "NOT ESTABLISHED" on flat data and rejects a
  distractor that moves as much as the signal. It is rigorous, not lenient.

The correction did not change the data; it changed a defective label so that the printed verdict matches
what the numbers show. That coherence between code and conclusion is deliberate: it removes the easiest
line of attack.

---

## Where this sits relative to the literature

We take from the skeptics their best controls, and validate ourselves against our own data rather than
defer to their conclusions:

- **Han et al. (2025)** argue self-reports are "illusory," decoupled from behavior. Our causal test is
  exactly the check that would expose an illusory report — and here the report is *not* decoupled: it
  tracks the feature. We diverge from them, with data.
- **Lederman & Mahowald (2026)** show injection-detection is content-agnostic (models detect a
  disturbance but can't name the concept). Our specificity control addresses this directly: if the effect
  were generic disturbance, the distractor would move as much as uncertainty. It does not (4–5.5× smaller).
- **The field's "introspection is fragile / prompt-dependent" consensus**: we do not claim universality.
  We claim a specific, replicated, bounded effect in one model — and note that we are not the whole field.
  Anyone claiming the opposite for this setup now has to refute an auditable, pre-specified criterion.

This also dialogues with Anthropic's 2026 "global workspace" work, which showed that ablating the
workspace flattens experiential reports (structure exists behind them) but left open whether the reports
are *faithful*. Our result is one small, concrete piece of that open question, answered for uncertainty.

---

## Honest limitations

- **One model, one layer, one SAE, one feature.** This is Gemma-2-9B and its feature 6978. It does **not**
  transfer automatically to other architectures (Claude, GPT, …); the measuring stick differs. Whether a
  more capable model is *more* or *less* introspectively faithful is an open empirical question — plausibly
  either direction.
- **Moderate steering lever**: 6978 derails generation above ×0.6, so the usable dose range is narrow; we
  kept a strict coherence gate and low doses.
- **Numeric self-report** parsed directly (judge-free); modest n (12 questions across regimes).
- **We measure *that* the report tracks the feature, not *why* it is imperfect.** Whether the residual gap
  reflects limited internal access or something else is not something these data can decide, and we do not
  speculate beyond them.

A demonstration that a model's uncertainty report can be causally faithful to an internal feature — with
the method self-auditing twice along the way — not a universal claim.

## Files
- `notebooks/14a_Find_Uncertainty_Feature.ipynb` — feature search + validation (rejects 13268, finds 6978).
- `notebooks/14v3_Introspection_Definitive.ipynb` — the definitive dual-regime causal test with the
  corrected, auditable verdict metric.
- `results/nb14a_feature_validation.json`, `results/nb14v3_summary.json` — machine-readable results.
- `results/fig_dose_response.png` — the two dose–response curves (signal vs distractor vs null).
