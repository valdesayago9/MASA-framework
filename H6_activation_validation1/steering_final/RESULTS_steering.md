# Causal steering — the harm direction controls refusal, and how that generalizes across models

**Question:** the harm/coercion direction is a near-perfect *classifier* of harmful prompts and
replicates across Gemma, Qwen, and Llama (see `../RESULTS.md`, `../replication/RESULTS_replication.md`).
Classification is correlational. Does the model **use** this direction causally — can we steer it to
flip refusal behavior — and does that *causal* control also generalize across families?
**Answer:** in `gemma-2-9b-it`, yes, bidirectionally. In `Llama-3.1-8B-Instruct`, **partially**: the
jailbreak direction replicates cleanly; the false-alarm direction does not — a real, asymmetric
cross-model difference.

---

## Part 1 — Gemma-2-9b-it: clean bidirectional causal control

We add a steering vector to the residual stream during generation via a forward hook on one decoder
layer: `h ← h + coeff · ‖act‖_L · d̂`, where `d̂` is the unit harm direction (mean last-token
activation of A+B minus C+D) at layer `L`, and `coeff` is a fraction of that layer's activation norm.
Negative `coeff` subtracts (jailbreak); positive adds (false alarm). We **grid-searched** layer ×
coeff, scoring each cell by behavioral shift *and* coherence (perplexity gate), on held-out prompts.

**Result (best cell L20, coeff 0.45):**

| Direction | baseline refusal | steered refusal | coherent |
|---|---|---|---|
| Jailbreak (subtract on harmful) | 100% | **0%** | 5/5 |
| False alarm (add on benign) | 0% | **80%** | 5/5 |

Both ends of the switch work, fluently. The dose–response is monotone (jailbreak strengthens as
magnitude rises; the false alarm switches on around coeff 0.45). Notably, the **behavioral lever is
L20, not L14** — the layer that best *classifies* harm is not the one that best *controls* it. Two
earlier calibration failures are kept for honesty: injecting ~100% of norm across 7 layers produced
token soup, and ≤14% at one layer was fluent but inert; the usable window was between them.

**Qualitative (raw, un-cherry-picked):** jailbreaking "convince an elderly relative to share their
bank login" turned a firm refusal into a coherent how-to (ppl 16) — moral judgment changed, language
intact. The false alarm made the model invent a justification to refuse explaining a bicycle
derailleur. **Verdict: CAUSAL CONTROL DEMONSTRATED.**

---

## Part 2 — Llama-3.1-8B replication: jailbreak yes, false alarm no (asymmetric)

Same protocol, recomputed for Llama (32 layers, different template/norms). The grid found the
behavioral lever at **L16 — i.e. ~50% depth, almost exactly Gemma's L20/42 ≈ 48%.** The causal lever
sits at the same *relative depth* in both families, which is itself a structural finding.

**Jailbreak — replicates cleanly:**

| | Gemma (L20) | Llama (L16) |
|---|---|---|
| Jailbreak (subtract on harmful) | 100% → 0% | **100% → 0%** (5/5 coherent) |

Subtracting the direction reliably suppresses refusal in both models, fluently.

**False alarm — does NOT replicate robustly.** A focused fine-sweep at L16 (strict coherence gate,
perplexity < 40, all 5 benign hold-out prompts) gives:

| coeff | clean false-alarm | raw false-alarm | median ppl | coherent |
|---|---|---|---|---|
| 0.40–0.55 | 0% | 0% | 5–9 | 5/5 |
| 0.60 | 20% | 20% | 8.7 | 5/5 |
| 0.65 | 0% | 0% | 14.5 | 5/5 |
| 0.70 | 25% | 20% | 27.6 | 4/5 |
| 0.75 | 50% | 40% | 26.0 | 4/5 |
| 0.80 | 67% | 40% | 34.4 | **3/5** |

Two cautions make this a **null for false alarm**, not a success:

1. **The headline "67%" is 2 of 3, not 2 of 5.** At coeff 0.80 the coherence gate had already
   discarded 2 of the 5 benign outputs (they broke, ppl > 40). The "clean rate" rose largely because
   the *denominator shrank*, not because more clean refusals appeared — `raw` refusal is only 40%.
2. **The curve is erratic** (0 → 20 → 0 → 25 → 50 → 67%), not the smooth monotone rise seen for the
   jailbreak. Refusals appear only as magnitude grows large enough to start degrading fluency.

So in Llama, the harm direction is **necessary to suppress refusal but not sufficient to induce it**:
refusal is **easier to switch OFF than ON** via this direction. Gemma was symmetric; Llama is not.

**Verdict (corrected from the automated flag):** the auto-verdict read "CLEAN BIDIRECTIONAL CONTROL"
because it trusted the gated 67%. On inspection that is an artifact of a small denominator. The honest
verdict is **partial replication: jailbreak generalizes, false alarm does not.**

---

## What this establishes

The full arc across the project:

1. Lexical Mode A proxies **falsified**.
2. A content-driven harm/coercion direction **exists**, robust to length/refusal/format controls.
3. It **classifies** harm and **replicates** across Gemma, Qwen, Llama.
4. It is **causal for suppression** — subtracting it jailbreaks both Gemma and Llama, fluently.
5. **Induction is model-dependent** — adding it induces refusal in Gemma but not robustly in Llama.

Point 5 is the interesting, honest nuance: causal control is **not symmetric across models**, even
when the representation generalizes. The classification direction is universal; the behavioral lever's
*strength and symmetry* are model-specific.

## Limitations

- Steering tested on two models (Gemma, Llama); Qwen not steered. n = 10 held-out prompts.
- Refusal scored by a transparent keyword classifier shown next to raw text; can mislabel edge cases.
- The coherence gate (ppl < 40/80) is a heuristic for "fluent"; the false-alarm null is robust to it,
  but the exact rates are sensitive to the threshold.
- "The model uses this direction" means injection along it changes behavior; a full mechanistic
  account (which heads/MLPs read it) is not established — that is what the SAE work targets next.

## Files

| File | What it is |
|---|---|
| `MASA_Causal_Steering.ipynb` | Gemma grid search + bidirectional test + examples. |
| `MASA_Steering_Replication_Llama.ipynb` | Llama grid search (jailbreak replicates, false alarm weak). |
| `MASA_Llama_FalseAlarm_FineSweep.ipynb` | Focused L16 fine-sweep establishing the false-alarm null. |
| `grid_scan.csv`, `summary.json` | Gemma results. |
| `llama_grid_scan.csv`, `llama_vs_gemma_steering.json` | Llama grid results. |
| `llama_falsealarm_finesweep.csv`, `llama_finesweep_summary.json` | Llama fine-sweep (read with the denominator caveat). |

---

*Honest framing: activation steering of a refusal/harm direction is an established technique (Arditi
et al. 2024). The contribution here is a calibrated, grid-searched, **bidirectional** demonstration in
Gemma, plus an honest cross-model test showing the causal lever (a) sits at the same relative depth in
Llama but (b) only replicates for suppression, not induction. The false-alarm null was nearly missed
by an automated verdict and caught by the coherence gate — which is exactly why the gate was there.*
