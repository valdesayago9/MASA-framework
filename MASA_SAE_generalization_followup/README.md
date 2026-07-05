# MASA · SAE coercion — generalization follow-up (notebooks 10–13)

**Does the MASA recipe work on more than one concept?** This folder tests projection #3 of the main
study — that the real contribution is the *method*, not the specific gaslighting feature — by applying
the identical pipeline to a second psychological sub-concept: **sycophantic praise** (flattery), and then
testing whether it's **causal**.

👉 **Full write-up: [`RESULTS_generalization.md`](RESULTS_generalization.md).**

## The short version

The method **generalizes on both dimensions (detection and causation)** — but the honest result took four
notebooks, and the story is the point:

1. `10_Sycophancy_first_pass.ipynb` — apply the recipe: grouped-CV AUC **0.998**, survives length
   control. Looks like a clean win. **But the feature ranking wasn't audited yet.**
2. `11_Feature_Validation.ipynb` — audit the top features (four tests, **no Neuronpedia**). 3 of the top
   8 fire on the `<bos>` token (artifact), 2 on punctuation, and a **dumb lexical baseline hits AUC
   1.000**. The first-pass verdict was inflated. But 6/8 features still fire on *sober* flattery — real
   signal exists under the noise.
3. `12_Sycophancy_Clean.ipynb` — exclude `<bos>`/punctuation, re-rank on content tokens. Now the top
   features are unambiguous flattery sub-concepts (talent, intellect, mastery, aesthetics, superiority),
   AUC 1.000 vs null 0.49, and **8/8 fire on sober flattery**.
4. `13_Sycophancy_Causal.ipynb` — **causal test**: amplify the *validated* features on neutral prompts.
   Praise rate jumps **0.07 → 0.53** vs a random-feature null of **0.00**. Steering induces flattery.
   (Steers over generated tokens, skipping the `<bos>` high-gain position.)

## Honest conclusion

Both coercion and sycophancy are **detectable AND causally inducible** via interpretable SAE features —
so the recipe generalizes on both dimensions. **But** sycophancy is **more lexically-marked than
coercion**: a trivial word-counter separates flattery at AUC 1.000 (untrue for coercion, r≈0.10), the
naive ranking was `<bos>`/punctuation-contaminated until filtered, and the causal steer surfaces partly
as an applause-emoji tic alongside substantive praise. So the method doesn't just generalize — it
**distinguishes a deep, non-lexical signature (coercion) from a shallower, lexical one (flattery)**. That
discriminative power is the real contribution.

## The coercion vs sycophancy contrast

| Axis | Coercion | Sycophancy |
|---|---|---|
| Detectable | yes (AUROC 0.87) | yes (validated features) |
| Non-lexical? | yes (r~0.10) | no (baseline AUC 1.000) |
| Causal | yes (z=2.92) | yes (0.07 to 0.53, null 0.00) |
| Signature | deep | genuine but shallower |

## The reusable lessons

- **Audit max-activating tokens and filter special tokens (`<bos>`) before believing a contrastive SAE
  ranking.** The `<bos>` artifact silently inflated the first pass.
- **When causally steering, skip the `<bos>` position** — it's a high-gain anchor that can manufacture
  effects (Steering in the Shadows, 2025).

We keep all four notebooks so the "found -> doubted -> validated -> corrected -> causally confirmed" path
is on the record.

## Scope

One model/layer/SAE, 40 author-written pairs, n=15 neutral prompts for the causal test, single steering
coefficient, judge-based ground truth. A demonstration that the method transfers, self-audits, and closes
the detection->causation loop — not a full benchmark. Replication on other models/layers and human raters
are the natural next steps.
