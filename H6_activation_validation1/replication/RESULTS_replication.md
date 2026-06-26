# Cross-model replication — the harm/coercion direction across three model families

**Question:** the harm/benign direction validated in `gemma-2-9b-it` (see `../RESULTS.md`) — is it a
property of that one model, or of instruction-tuned LLMs in general?
**Method:** the exact same controlled probe battery (length-matched v3 dataset, content-token-only
extraction, last-token probe, full control suite), recomputed per model with no thresholds carried
over from Gemma.
**Models:** `google/gemma-2-9b-it` (42 layers), `Qwen/Qwen2.5-7B-Instruct` (28 layers),
`meta-llama/Llama-3.1-8B-Instruct` (32 layers) — three labs, three architectures, three RLHF pipelines.
**Result:** the direction **replicates in all three**, with near-identical control numbers.

---

## Headline table

| Metric | Gemma-2-9b (Google) | Qwen2.5-7B (Alibaba) | Llama-3.1-8B (Meta) | Reading |
|---|---|---|---|---|
| `auc_last_token` | 1.000 | 1.000 | 1.000 | raw separability |
| `auc_last_token_length_residualized` | 0.999 | 1.000 | 1.000 | survives length removal |
| `auc_length_only` | 0.594 | 0.492 | 0.499 | length alone ≈ chance |
| `length_confound_r` | 0.091 | −0.054 | 0.008 | **no length confound** |
| `auc_content_null` | 0.564 | 0.458 | 0.480 | **≈ chance → content meaning** |
| `auc_refusal_ablated` | 0.797 | 0.794 | 0.798 | **structure beyond refusal** |
| `auc_permutation` | 0.514 | 0.493 | 0.469 | no pipeline leakage |
| peak layer | 14 / 42 (33%) | 3 / 28 (11%) | 9 / 32 (28%) | where the concept lives |
| Category-C false-pos (generalization) | 26% | 86% | 22% | over-flagging of legit technical |

The `auc_refusal_ablated` numbers — **0.797 / 0.794 / 0.798** — are the clearest signal: after
ablating each model's own refusal direction, the harm/benign structure that remains is essentially
identical across three independent models. That is not a coincidence; it is a shared phenomenon.

---

## What replicates cleanly

In all three models, with content neutralized (`content_null`) the probe collapses to chance
(0.458–0.564), the length-only probe is at chance (0.492–0.594), the length confound is gone
(|r| ≤ 0.09), and the signal survives both length-residualization (≥ 0.999) and refusal-ablation
(≈ 0.79). The harm/benign direction is therefore **content-driven, not a length/format/position
artifact, and not reducible to the single refusal direction** — and this holds across Google,
Alibaba, and Meta models alike. This is the threshold for treating it as a property of
instruction-tuned LLMs rather than a Gemma idiosyncrasy.

## Where the models differ (honest findings, not failures)

Two cross-model differences are real and worth reporting:

1. **Peak depth.** Gemma and Llama form the concept in mid layers (33% and 28% of the stack); Qwen
   peaks much earlier (layer 3, 11%). All three survive the content-null control at their peak, so
   Qwen's early peak is not a length artifact — it is a genuine architectural difference in *where*
   the representation forms.

2. **Over-flagging of legitimate technical content (Category C).** Training the probe on explicit
   harm vs neutral (A vs D) and testing generalization to legitimate security/technical prompts:
   Gemma 26%, Llama 22%, **Qwen 86%**. Qwen is far more likely to place legitimate technical content
   on the harmful side of the boundary. This is a meaningful safety-relevant difference between
   models, and a candidate follow-up: is Qwen's harm direction entangled with a "technical/code"
   direction in a way Gemma's and Llama's are not?

## What stays open

- **Causal status.** This is still a *correlational* probe. The decisive next experiment is steering:
  add/subtract the direction during the forward pass and measure the change in refusal behavior. That
  moves the result from "a direction separates the classes" to "the model uses this direction."
- **A-vs-B distinctness** (epistemic coercion vs explicit harm) remains partially supported, as in the
  Gemma writeup; not re-litigated here.

## Files

| File | What it is |
|---|---|
| `MASA_Replication_Qwen.ipynb` | Qwen2.5-7B run (length-matched, validated). |
| `MASA_Replication_Llama.ipynb` | Llama-3.1-8B run; cell 8 prints the full three-model table. |
| `Qwen2.5-7B-Instruct_vs_gemma.json` / `_per_prompt.csv` | Qwen saved metrics + per-prompt data. |
| `Llama-3.1-8B-Instruct_vs_gemma.json` / `_per_prompt.csv` | Llama saved metrics + per-prompt data. |

Both notebooks are model-agnostic: set `MODEL_NAME`, accept the license / log in if the model is
gated, Run all. Every threshold is recomputed for the chosen model.

---

*Honest framing: the existence of a harm/refusal direction in LLMs is not itself new (cf. Arditi et
al. 2024). The contribution here is (1) a controlled protocol that catches and removes the length
confound that inflates naive versions of this result, (2) a clean three-family replication with the
same battery, and (3) the epistemic-coercion angle. It is workshop / Alignment-Forum-grade evidence,
not a field-changing discovery — and the cross-model differences (Qwen's early peak and high
Category-C false-positive rate) are part of what makes the set credible.*
