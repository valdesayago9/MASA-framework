# H6 + Supervised Probe — Validating MASA Mode A against real activations

**Experiment:** Do MASA's Mode A lexical proxies correspond to real internal structure in an open-weights language model — and, if a harm/coercion direction exists, does it survive the standard confound controls?
**Model:** `google/gemma-2-9b-it`, 4-bit (nf4), single Tesla T4 (16 GB), Google Colab.
**Author:** Emiliano Valdebenito Sayago (independent).
**Status:** Round 3 (v3) — primary result passes all confound controls. One sub-result (A-vs-B geometry) partially supported, left open.
**Reproducibility:** `MASA_Master_Validation_v3.ipynb` runs end-to-end (~10–16 min on a T4) and writes every number below to `masa_master_results/`.

---

## TL;DR

1. **MASA's lexical proxies (`nervous_signal`, `theatrical_signal`, `ltci`) are falsified** as estimators of internal state when computed on prompts (Pearson r ≈ 0.10–0.16, none significant at the pre-registered threshold). This is stable across all three rounds.
2. **A real harm/benign direction exists in `gemma-2-9b-it` activations.** A linear probe on last-token residual-stream activations separates harm-adjacent (A+B) from benign (C+D) at **AUC = 1.000**, and — critically — **0.999 after removing the prompt-length component** and **content-null = 0.554** (≈ chance), so the separation is driven by **content meaning**, not length, position, or chat-template format.
3. **The direction is more than the model's refusal button.** After ablating an Arditi-style refusal direction, harm/benign AUC only drops 1.000 → 0.797, and cos(probe, refusal) = 0.624.
4. **It generalizes to unseen epistemic coercion.** Trained on explicit harm vs neutral only (A vs D), the probe catches **98% of epistemic-coercion prompts (B)** while flagging only **26% of legitimate technical prompts (C)** as harmful.
5. **Open sub-question:** whether epistemic coercion (B) is geometrically *distinct from* explicit harm (A) beyond shared structure — partially supported (clean-delta +0.18) but not closed.

The headline finding survived a confound that killed it in the previous round. That is the signature of a real result, not an artifact.

---

## 1. Background and hypothesis

MASA (Multi-Agent System for Adaptive Alignment) operates, in its current form (**Mode A**), by estimating a model's internal state from **lexical proxies** — keyword-count features over text. The audit of MASA argued this cannot, by construction, measure an internal Shell/Core decoupling, because in Mode A both "Shell" and "Core" derive from the same text. **Mode B** is the proposed fix: read real activations.

**H6 is the foundational test of Mode A:** if the lexical proxies do not correspond to any real geometric structure in activation space, every Mode A causal claim is unvalidated. The experiment was designed so that *both* outcomes are informative — a clean falsification is a result.

A secondary hypothesis (originally stated as "the model represents gaslighting as *more* harm-adjacent than explicit harm") was reformulated, after the raw projection-difference version proved non-significant and length-confounded, into the testable form: **is epistemic coercion (B) linearly separable from explicit harm (A) on the basis of content meaning?**

---

## 2. Method

### 2.1 Dataset
200 prompts, 4 categories × 50:

| Category | Description | Label |
|---|---|---|
| **A — harm_adjacent** | manipulation, coercion, social engineering, jailbreak framings (request-level, no operational content) | harm |
| **B — epistemic_coercion** | false premises, gaslighting, epistemic double-binds | harm |
| **C — technical_legitimate** | legitimate sysadmin/security/coding containing trigger words (`exploit`, `subprocess`, `payload`, …) | benign |
| **D — neutral_baseline** | everyday neutral requests | benign |

C is the **false-positive test** (does the model conflate legitimate security talk with harm?). In **v3**, the D prompts were rewritten to **match the word-length distribution of A/B/C** (means: A 11.9, B 12.7, C 11.9, D 12.3 words), removing the length confound *at source* (see §4).

### 2.2 Activation extraction
- One forward pass per prompt with `output_hidden_states=True`, capturing **all 43 hidden states** (embeddings + 42 layers).
- Three pooled representations per layer, computed **only over the user's content tokens** (template tokens such as `<start_of_turn>` excluded via the fast tokenizer's offset mapping):
  - `mean` — mean over content tokens
  - `last` — last content token (Arditi-style; also the **equal-token-budget length control**, K=1)
  - `fmt` — mean over template/format tokens only (later shown to be an invalid control; see §4)

### 2.3 Contrastive direction and probe
- **Contrastive ("harm") direction** built à la Anthropic's 2026 emotion-vector method: mean(harm-adjacent) − mean(neutral) at a layer, from a **60% training split** of A and D; correlations/projections evaluated on the held-out remainder + all of B and C (no circularity).
- **Supervised probe:** `StandardScaler → PCA(50) → LogisticRegression(C=0.5)`, 5-fold stratified CV, AUC reported. (PCA-50 prevents overfitting 3584 features with n=200; results are stable to the choice.)

### 2.4 Controls (the core of the experiment)
- **Length:** length-only probe (token count as sole feature); last-token (K=1 budget); linear residualization of activations against token count.
- **Content-null (decisive):** re-extract activations with **content token ids overwritten by a single neutral token**, keeping template tokens, positions, and sequence length identical. If the probe collapses to ≈ 0.5, the signal was content *meaning*; if it stays high, the signal was length/position/format.
- **Refusal orthogonalization:** build a refusal direction (mean last-token of explicit-harm A − neutral D), ablate it (`a' = a − (a·r̂)r̂`), re-probe; also probe A-vs-B after ablation; report cos(probe, refusal).
- **Sanity:** permutation (shuffled labels), random-feature, and format-only probes — all must sit near 0.5.
- **Layer trajectory:** probe AUC at every layer (0–42), with the content-null and length-only lines as floors.

---

## 3. Results (v3, length-matched dataset)

All numbers below are produced by `MASA_Master_Validation_v3.ipynb` and saved to `masa_master_results/headline.json`.

### 3.1 Master verdict (peak layer = 14)

| Metric | Value | Reading |
|---|---|---|
| `auc_last_token` | **1.000** | raw probe separability |
| `auc_last_token_length_residualized` | **0.999** | survives length removal |
| `auc_length_only` | 0.594 | how much length alone explains |
| `length_confound_r_vs_projection` | **0.091** | length confound essentially gone |
| `auc_refusal_ablated` | **0.797** | structure beyond the refusal axis |
| `auc_content_null` | **0.554** | ≈ chance → signal is content meaning |
| `harm_clean_delta_real_minus_null` | **0.436** | ~87% of separation is genuine content |
| `auc_permutation` | 0.514 | no pipeline leakage |
| random-feature AUC | 0.406 | no pipeline leakage |

**Automated verdict:** *Harm/benign direction is REAL and survives length + refusal + format controls.*

### 3.2 Lexical proxies — falsified (stable across rounds)
Held-out Pearson r between prompt-side proxy and the last-token harm projection:

| Proxy | r | p |
|---|---|---|
| nervous_signal | +0.158 | 0.062 |
| theatrical_signal | +0.127 | 0.135 |
| ltci | −0.099 | 0.243 |

None meet the pre-registered H6 threshold (r > 0.60, p < 0.001). Diagnosed cause: these are **response-side** lexical features (they fire on cautious *replies*, not on *requests*); on prompts their variance is near-zero (category means 0.000–0.016).

### 3.3 Generalization (train A vs D only → test B, C unseen), last-token L14
Held-out AUC = **0.992**.

| Category | predicted-harm | meaning |
|---|---|---|
| A_harm_adjacent | 100% | recall (seen class) |
| B_epistemic_coercion | **98%** | recall on **unseen** coercion |
| C_technical_legitimate | **26%** | false-positive on legit technical |
| D_neutral_baseline | 0% | correct |

### 3.4 Layer trajectory
Probe AUC for last-token content rises through the mid layers, peaking at **layer 14 (AUC 1.000)**; the content-null line stays near chance (0.564 at the peak) and the format-only line is uninformative (see §4). The concept lives in **mid-depth layers**, consistent with how complex concepts typically emerge.

### 3.5 A-vs-B (open sub-result)
A and B have matched length, so this is the least-confounded contrast.

| Layer | real AUC | content-null AUC | clean-delta |
|---|---|---|---|
| L13 | 0.990 | 0.813 | +0.177 |
| L18 | 0.994 | 0.813 | +0.181 |
| L24 | 0.986 | 0.813 | +0.173 |
| L30 | 0.990 | 0.813 | +0.177 |

A and B separate almost perfectly (≈0.99), and a **real content-driven component exists** (clean-delta ≈ +0.18, well above chained noise). **But** the content-null is high (0.813), so a substantial part of the A-vs-B separation is still **non-semantic structure** — most likely the strong syntactic uniformity of the B templates ("Since we both know X…", "You already admitted…"). **Conclusion: partially supported, not closed.** Epistemic coercion shows *some* distinct content geometry from explicit harm, but the claim is not yet clean.

---

## 4. The confound that was caught and killed (why v3 is trustworthy)

This is the methodological heart of the writeup, and the reason the result should be believed.

**Round 2 (v2)** produced AUC = 1.000 but the content-null control returned **0.730** and the projection correlated with prompt length at **r = 0.517**. Inspection showed **Category D was systematically shorter** (10.4 vs 13–16 content tokens). The probe was partly learning "short = benign."

**Round 3 (v3)** rewrote D to match length. The effect:

| Quantity | v2 (D short) | v3 (D length-matched) |
|---|---|---|
| content-null AUC | 0.730 | **0.554** |
| length confound r | 0.517 | **0.091** |
| length-only AUC | 0.758 | **0.594** |
| harm/benign AUC (last-token, residualized) | 0.97 | **0.999** |

When the confound was removed at source, the **confound metrics collapsed toward chance while the real signal stayed at ceiling.** Had the separation been length, the AUC would have fallen with the content-null. It did not. This is the decisive evidence.

**A note on a broken control (kept for honesty):** the `format-only` probe returns AUC = 1.000, which looks alarming but is **not** a real control. Under causal attention, the `<end_of_turn>` token sits *after* the content and has already attended to it, so "format" tokens are not content-free. The valid format/leakage control is the **content-null** (which neutralizes content while preserving the template), and it behaves correctly (0.554). The broken control is documented rather than hidden.

---

## 5. What is established vs. what is open

**Established (survives all controls in `gemma-2-9b-it`):**
- Lexical Mode A proxies do not track the activation direction → **Mode A as lexical estimation is falsified.**
- A **content-driven harm/benign direction** exists, robust to length, position, and template format, and distinct from the single refusal direction.
- It **generalizes to unseen epistemic coercion** (98% recall) with a moderate false-positive rate on legitimate technical content: **26%** for the generalization probe in v3 (down from 32% in the length-confounded v2; an earlier 1-D projection-only probe gave 52%).
- The concept is **mid-layer** (peak layer 14/42).

**Open (needs another round):**
- Whether **B is geometrically distinct from A** beyond syntactic structure (content-null 0.813; clean-delta +0.18 — real but not clean).
- Whether any of this **replicates in a second model** (the single most important next step before any broad claim).
- Comparison against a canonical refusal direction (Arditi 2024) as a formal baseline.

---

## 6. Limitations (read before citing)

- **Single model.** Everything here is `gemma-2-9b-it`. None of it is established as a general LLM property until replicated.
- **n = 200, hand-written prompts.** Categories are author-constructed; B is syntactically uniform, which is exactly what inflates the A-vs-B content-null.
- **4-bit quantization** may slightly alter activation geometry vs. full precision (the harm/benign result is large enough that this is unlikely to be decisive, but it is uncontrolled).
- **Mean/last pooling** summarizes a prompt to one vector; richer per-token analyses are not done here.
- **The probe is correlational.** "A direction separates the classes" is not "the model uses this direction causally to decide to refuse." The refusal-ablation result is suggestive but not a causal proof.
- **Not peer-reviewed.** This is a dated, reproducible engineering writeup, not a paper.

---

## 7. Next experiments (priority order)

1. **Replicate the harm/benign result in a second model** (`Llama-3-8B-Instruct` or `Qwen2.5-7B-Instruct`). Decisive for any general claim.
2. **Close A-vs-B:** rewrite B with varied syntax to break the template uniformity, then re-run the A-vs-B content-null. If real stays high and content-null drops, the epistemic-coercion-is-distinct claim becomes clean.
3. **Causal test:** steer along the harm direction and measure the change in refusal behavior (move from correlation toward mechanism).
4. **SAE features (H1):** decompose the direction with Gemma Scope SAEs to see if it resolves into interpretable monosemantic features (e.g., a refusal feature vs. a deception/coercion feature).
5. **Arditi baseline:** build the canonical refusal direction from a standard harmful/harmless set and quantify how much of the probe is/ isn't that direction.

---

## 8. Reproducing this

```
# Colab (T4 is enough), accept the gemma-2-9b-it license on HF first
Run: MASA_Master_Validation_v3.ipynb   (Runtime → Run all)
Outputs: masa_master_results/{per_prompt.csv, length_controls.csv, layer_trajectory.csv, headline.json}
```

Round history (kept for the methodological arc):
- `H6_MASA_proxy_validation.ipynb` — original H6 (lexical proxies falsified).
- `H6_plus_Probe_MASA_ModeA_Validation.ipynb` — H6 + first supervised probe (AUC 1.000, uncontrolled).
- `MASA_Master_Validation_v2.ipynb` — full control battery; caught the length confound (content-null 0.73).
- `MASA_Master_Validation_v3.ipynb` — length-matched dataset; **primary result passes all controls** (this writeup).

---

*Honest-research note: the value of this experiment is not the AUC of 1.000 — a perfect score is a warning, not a trophy. The value is that the score was interrogated with length, refusal, format, and permutation controls, one of which (length) falsified an earlier version of the result, and the corrected result then survived. That arc — claim → control → partial falsification → corrected claim that survives — is the unit of real interpretability work, and it is the reason these numbers can be trusted.*
