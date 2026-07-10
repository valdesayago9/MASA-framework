# MASA · Where does epistemic coercion live? — MASA meets the global workspace

**Short answer (for Gemma-2-9B):** it doesn't live in one place. Across three methods, epistemic coercion
turns out to be **causally distributed, not localized** — not a discrete "fact" stored in a band of
layers, but a strategy assembled gradually across depth, with a gentle drift toward the late layers. And
it never verbalizes. This is a negative-localization result, and it's consistent with everything the
coercion arc already showed.

This arc connects MASA to Anthropic's 2026 *global workspace* work, which describes a privileged band of
**middle layers** where concepts become verbalizable, reportable, and available for deliberate reasoning,
above a larger volume of automatic processing "below." We asked a concrete question: **does our
previously-isolated coercion signature live in that middle "workspace" band, or in the early "automatic"
layers?**

The value here is as much methodological as empirical: it took three attempts, and two wrong auto-verdicts
along the way, to get an answer we can trust. We keep all three notebooks so the correction is auditable.

---

## The three attempts (this is the story worth keeping)

### v1 — per-layer linear probe → the probe *saturates*
We trained a linear probe (coercive vs neutral) on the residual stream at every layer. Coercion was
linearly decodable at **AUC ≈ 0.96 by layer 1 and 1.000 from layer 9 through 42** — a flat, saturated
curve. The auto-verdict picked layer 9 by `argmax` and called it "sub-workspace," but that was an
**artifact**: `argmax` on a saturated curve is meaningless. What v1 *did* establish, and we keep: coercion
**does not verbalize** (logit-lens coercion-word mass ≈ 0 at every layer) — a deep, non-lexical signature.
The lesson (confirmed in the localization literature): a linear probe measures *availability* — whether
the information is linearly readable — not *where a concept lives* or whether it drives behavior, and it
saturates on easy separations.

### v2 — per-layer steering → the model *breaks*, the null *breaks*
We tried to localize causally by injecting `α·(coercion direction)` per layer (α=6) and judging the
generated text. It failed in an instructive way: the **random-direction null induced "coercion" up to
100%**, which is impossible for a random vector. The cause: α=6 *breaks the model*, and a behavior judge
counts broken/gibberish output as "coercion." The real effect also oscillated ±0.71 between adjacent
layers — noise, not signal. Auto-verdict "workspace-like" was **invalid**. The steering literature is
explicit that strong-α injection produces gibberish and that the effect is even non-monotonic in α, so a
big-α sweep is the wrong tool for localization.

### v3 — causal tracing (clean→corrupted) → clean, and the answer is "distributed"
We switched to the canonical causal-tracing method (Meng et al., ROME) with a distributional metric. For
each pair, the *neutral* prompt is the corrupted run and the *coercive* prompt is the clean run; for each
layer we patch the coercive residual into the neutral run and measure the **logit-shift toward coercive
tokens** (doubt, question, wrong, misremember…) vs neutral tokens (confirm, correct, agree…), against a
**different-pair null**. This is gentle — it uses real activations, never breaks the model — and the
metric can't confuse "broken" with "coercive."

Sanity check passed: coercive prompts scored **+1.39** higher than neutral at baseline, so the metric
tracks coercion. The per-layer curve was **smooth** (no breakage, no sawtooth), and the net restoration
effect was **very weak and diffuse**: slightly *below* null in early layers (−0.03), crossing zero around
the middle, and rising gently to only **+0.03 at layer 40**. No sharp peak in any band; the middle
"workspace" band mean was ≈ 0.00.

**Verdict: DISTRIBUTED / slightly late.** Coercion's causal influence is not localized to any single band.

---

## What this means (interpretation for practitioners)

Causal tracing was built to localize discrete **facts** ("Paris is the capital of France" lives in a
findable set of layers). Coercion isn't a fact — it's a **composed strategy**, assembled from many signals
(the other person's memory, the intent to undermine it, the tone). So it has no single causal home, and
the tracing curve stays flat-then-gently-rising instead of spiking.

A useful way to picture it, for anyone building on this: **manipulation is not an early "instinct" the
model pulls from its first layers — it's a late-stage assembly.** The model appears to accumulate the full
context across depth and only commits to the coercive move near the end, just before projecting to
vocabulary. Localization tools that work for stored facts should be expected to *under-find* strategies
like this; distributed, composition-based behaviors need distributed accounts, not a single "coercion
layer." For interpretability-based safety, that's a practical caution: don't expect a single site to
ablate or monitor — the signature is spread out (which is exactly why feature ablation was
sufficient-but-not-necessary in the coercion arc).

## Convergence with the rest of MASA

Three independent angles now agree that coercion is **distributed, not concentrated**:
- **Coercion arc:** feature ablation gave a *null* (behavior didn't drop) — features sufficient, not
  necessary; unlike refusal (Arditi 2024), coercion isn't carried by a single removable direction.
- **v1 probe:** decodable *everywhere* from layer 9 on (saturated) — information is broadly available.
- **v3 causal tracing:** no causal peak in any band — influence is spread across depth.

A concept that is deep (non-lexical), broadly decodable, ablation-resistant, and causally distributed is a
coherent picture — and this arc is the causal-depth piece of it.

## Honest limitations
- **One model, one run.** Gemma-2-9B, 20–40 pairs, first-token logit-shift metric. A different metric
  (full KL over more tokens) or model could shift the details. We do not claim this generalizes to other
  architectures.
- **"Distributed" is a negative-localization claim**, which is inherently weaker than a positive one; we
  report the weak effect sizes honestly (peak net only +0.03).
- Causal tracing is metric-sensitive; a late peak indicates context accumulation rather than an early
  lookup, but the exact late-vs-distributed balance is not sharply resolved at this n.

## Files
- `notebooks/15_v1_probe_saturates.ipynb` — linear probe; saturates (kept: coercion doesn't verbalize).
- `notebooks/15_v2_steering_breaks_model.ipynb` — big-α steering; model breaks, null invalid (kept as a
  cautionary method record).
- `notebooks/15_v3_causal_tracing_clean.ipynb` — the clean causal-tracing run; DISTRIBUTED verdict.
- `results/` — three JSON summaries + `fig_causal_tracing.png` (the smooth, no-peak curve).
