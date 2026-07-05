# MASA · SAE coercion — generalization follow-up (notebooks 10–12)

**Does the MASA recipe work on more than one concept?** This folder tests projection #3 of the main
study — that the real contribution is the *method*, not the specific gaslighting feature — by applying
the identical pipeline to a second psychological sub-concept: **sycophantic praise** (flattery).

👉 **Full write-up: [`RESULTS_generalization.md`](RESULTS_generalization.md).**

## The short version

The method **generalizes** — but the honest result took three notebooks, and the story is the point:

1. `10_Sycophancy_first_pass.ipynb` — apply the recipe: grouped-CV AUC **0.998**, survives length
   control. Looks like a clean win. **But the feature ranking wasn't audited yet.**
2. `11_Feature_Validation.ipynb` — audit the top features (four tests, **no Neuronpedia**). Uh-oh: 3 of
   the top 8 fire on the `<bos>` token (artifact), 2 on punctuation, and a **dumb lexical baseline hits
   AUC 1.000**. The first-pass verdict was inflated. But 6/8 features still fire on *sober* flattery, so
   real signal exists under the noise.
3. `12_Sycophancy_Clean.ipynb` — exclude `<bos>`/punctuation, re-rank on content tokens. Now the top
   features are unambiguous flattery sub-concepts (talent, intellect, mastery, aesthetics, superiority),
   AUC 1.000 vs null 0.49, and **8/8 fire on sober flattery**.

## Honest conclusion

The recipe generalizes: the same pipeline found genuine, interpretable flattery features in a second
concept. **But** sycophancy is **more lexically-marked than coercion** — a trivial word-counter reaches
AUC 1.000 (which was *not* true for coercion, where keyword proxies gave r≈0.10). So the method not only
generalizes, it **distinguishes a deep, non-lexical signature (coercion) from a shallower, lexical one
(flattery)** — a useful property, and a caution against trusting any high AUC before auditing the
features.

## The reusable lesson

**Audit max-activating tokens and filter special tokens (`<bos>`) before believing a contrastive SAE
ranking.** The `<bos>` artifact silently inflated the first-pass ranking; only per-token inspection
caught it. We keep all three notebooks so the "found → doubted → validated → corrected" path is on the
record.

## Scope

One model/layer/SAE, 40 author-written pairs, correlational (no causal steering test for sycophancy
yet). A demonstration that the method transfers and self-audits, not a full benchmark. Replication and a
causal test are the natural next steps.
