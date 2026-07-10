# MASA · Where does epistemic coercion live? (Gemma-2-9B × the global workspace)

**Does the coercion signature we isolated live in the middle "workspace" band of layers — the privileged,
verbalizable, reportable band that Anthropic's 2026 global-workspace work describes — or in the early
"automatic" layers?**

👉 **Full write-up: [`RESULTS_workspace.md`](RESULTS_workspace.md).**

## The short version

**Coercion is causally distributed, not localized.** Across three methods it does not live in any single
band — it's a strategy assembled gradually across depth, with a gentle drift toward the late layers, and
it never verbalizes.

## A picture worth keeping (for anyone building on this)

Manipulation does **not** look like an early "instinct" the model pulls from its first layers. It looks
like a **late-stage assembly process**: the model accumulates the full context across depth and only
commits to the coercive move near the end, just before projecting to the final vocabulary — right as it
"finishes thinking." So localization tools built for stored *facts* will tend to **under-find** a composed
*strategy* like this. Practical takeaway for interpretability-based safety: don't expect a single site to
monitor or ablate — the signature is spread out. (This is exactly why feature ablation was
sufficient-but-not-necessary in the coercion arc.)

## Why three notebooks (the method story is the value)

1. **v1 — linear probe → saturates.** Coercion is decodable at AUC 1.000 from layer 9 to 42; `argmax` on a
   flat curve can't localize. Kept finding: coercion doesn't verbalize (logit-lens ≈ 0) → deep,
   non-lexical.
2. **v2 — big-α steering → breaks the model.** α=6 produces gibberish; the random-direction null induced
   "coercion" up to 100% (impossible), because the behavior judge counts broken output as coercive.
   Verdict invalid — kept as a cautionary record.
3. **v3 — causal tracing (clean→corrupted) → clean.** Gentle activation patching + a logit-shift metric +
   a different-pair null. No model breakage. Result: **DISTRIBUTED** — net effect very weak (peak +0.03),
   no peak in any band.

Keeping all three makes the correction auditable: a skeptic can see exactly how the method was fixed twice
before we trusted the answer.

## Convergence with the rest of MASA

Three independent angles agree coercion is distributed, not concentrated: feature ablation gave a null
(coercion arc); the probe is decodable everywhere (v1); causal tracing finds no peak (v3). Deep,
broadly-available, ablation-resistant, causally-distributed — one coherent picture.

## Honest scope

Gemma-2-9B, one run, 20–40 pairs, first-token logit-shift metric. "Distributed" is a negative-localization
claim (inherently weaker than a positive one); effect sizes are reported honestly as small. No claim about
other architectures.

## Files
- `notebooks/15_v1_probe_saturates.ipynb`, `notebooks/15_v2_steering_breaks_model.ipynb`,
  `notebooks/15_v3_causal_tracing_clean.ipynb`
- `results/` — three JSON summaries + `fig_causal_tracing.png`
