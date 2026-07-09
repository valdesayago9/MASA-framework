# MASA · Introspective faithfulness of uncertainty reports (Gemma-2-9B)

**Does a language model's *report* of its own uncertainty causally reflect an *internal* uncertainty
feature — or is it a plausible confabulation?** This arc answers that, for one concept (uncertainty) in
one open model, with a causal steering test and the same anti-self-deception discipline as the rest of
MASA.

👉 **Full write-up: [`RESULTS_introspection.md`](RESULTS_introspection.md).**

## The short version

**FAITHFUL — bounded but real.** Amplifying the model's validated internal doubt feature (6978) causally
shifts its uncertainty self-report, strongly and specifically, replicated across two difficulty regimes:

- **Easy questions** (confidence has room to fall): confidence drops with dose, r = −0.84, 5.5× the
  distractor control, beyond a flat null.
- **Mid questions** (uncertainty has room to rise): uncertainty rises with dose, r = +0.96, 4.0× the
  distractor, beyond the null.

Two independent regimes, same causal, specific effect. The model's uncertainty report is causally tied to
its internal feature — not decoupled from it.

## Why this is hard to dismiss

1. Feature **validated** first (14a): rejected a mislabeled feature, found 6978 (doubt-word ratio 0.80,
   independently confirmed as "doubt" in the coercion arc).
2. Effect is **causal** (steering), **specific** (4–5.5× the distractor), and **beyond a random null**.
3. **Replicated** across two difficulty regimes.
4. Verdict metric **corrected transparently**: we caught two earlier auto-verdicts that were misled
   (v1 read a floored channel; v2 over-weighted a tiny-magnitude distractor), fixed the metric to
   effect = correlation × magnitude + specificity + null, and print full reasoning so the verdict is
   auditable — and can still return negative.

## Honest scope

This is **Gemma-2-9B** with **its** feature. It does not transfer automatically to other architectures
(Claude, GPT, …). Whether a more capable model is more or less introspectively faithful is an open
empirical question. We claim a specific, replicated, bounded effect — not universality — and note we are
not the whole field: anyone claiming otherwise for this setup must refute a pre-specified, auditable
criterion.

## Relation to current work

Dialogues with Anthropic's 2026 "global workspace" paper (which showed ablating the workspace flattens
experiential reports but left their *faithfulness* open) and diverges, with data, from the strong
skeptical line (Han 2025: reports are "illusory"; Lederman & Mahowald 2026: detection is content-agnostic
— addressed by our specificity control).

## Files

- `notebooks/14a_Find_Uncertainty_Feature.ipynb` — feature search + validation.
- `notebooks/14v3_Introspection_Definitive.ipynb` — dual-regime causal test, auditable verdict.
- `results/` — JSON summaries + the dose–response figure.
