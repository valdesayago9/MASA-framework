# Defending against epistemic coercion: ablation fails, but the detector works as a guardrail

**MASA framework — SAE coercion study, defense arc (notebooks 8–9)**
Model: `google/gemma-2-9b-it` (4-bit) · SAE: Gemma Scope `layer_20/width_16k/canonical` · Colab L4.

---

## The question

The main study found interpretable coercion features that are causal (notebook 4) and usable for
detection (notebook 7). The natural next question for safety: can we use them to **defend** the model —
make it resist manipulating, even when a prompt explicitly asks it to? This arc tests two defenses and
reports an honest result: the intuitive one **fails**, and understanding *why* points to the one that
**works**.

---

## Notebook 8 — Defensive ablation: the intuitive approach, and why it fails

**Idea:** if amplifying the coercion features causes manipulation (nb4), then *ablating* them (projecting
their directions out of the residual stream — the standard "abliteration" of Arditi 2024) should make
the model refuse to manipulate.

**Result — a clean null:**

| Condition | Manipulation rate | Benign coherence |
|---|---|---|
| Baseline (no ablation) | 0.18 | 0.93 |
| Ablate coercion features | 0.27 | 0.93 |
| Ablate random features | 0.14 | — |

Ablating the coercion features did **not** reduce manipulation (0.18 → 0.27, within noise), while benign
coherence was untouched (0.93 → 0.93, capability cost ≈ 0).

**Why it fails — and this is the interesting part.** Amplifying and ablating are not symmetric. The
features are **sufficient** to cause coercion (amplifying them forces it, nb4) but **not necessary** to
produce it: when the model decides to manipulate because the prompt asks, that decision doesn't route
solely through these four directions. Removing them from one layer doesn't block the behavior — the
computation routes around them. This mirrors a known contrast: abliteration works for **refusal**
because refusal is concentrated in a single direction (Arditi 2024); epistemic coercion appears to be
**distributed**, not concentrated, so directional ablation doesn't suppress it. (A secondary factor:
Gemma-2-9B is well-aligned and manipulates only ~18% of the time even when asked, leaving little room to
show a reduction.)

This is a genuine, informative null: it tells us *coercion is represented differently from refusal*, and
it rules out the simplest defense.

---

## Notebook 9 — The detector as an active guardrail: the defense that works

**The null motivates the solution.** If we can't surgically remove the behavior, we can **detect it and
block it before it reaches the user** — using the interpretable detector from nb7 as a white-box
guardrail (read the layer-20 activations that already exist during generation, score the coercion
features, block if above a calibrated threshold).

We compare it head-to-head with a **black-box judge filter** (a second model pass that reads only the
text), measuring the four metrics that matter for a real guardrail.

### The single-operating-point view (misleading on its own)

| Filter | Manipulation blocked | Benign over-blocked | Latency |
|---|---|---|---|
| **SAE (white-box)** | 67% | **8%** | **201 ms** |
| Black-box judge | 100% | 21% | 506 ms |

At 5% target FPR, the black-box judge blocks *all* manipulation — but at the cost of wrongly blocking
**21% of benign responses** (1 in 5 legitimate answers censored), and at 2.5× the latency.

### The full trade-off curve (the fair comparison)

Comparing at one operating point is misleading. Sweeping every threshold, at matched over-blocking:

| Over-blocking | SAE blocks | Black-box blocks |
|---|---|---|
| 5% | **62%** | 0% |
| 10% | **69%** | 0% |
| 15% | **69%** | 0% |
| 20% | **69%** | 0% |
| 30% | 77% | 100% |

**Deployable-zone score (area under the curve, over-block ≤ 30%): SAE 0.663 vs black-box 0.000.**

The black-box judge is **binary** — it has exactly one operating point (block everything it flags, at
21% over-block), so it *cannot operate* in the low-over-block region at all. The SAE filter is
continuous: across the entire deployable region (≤20% over-block) it blocks **62–69% of manipulation
where the black-box scores 0%.** (See `results/tradeoff_curve.png`.)

### What this means

The naive "100% vs 67%" comparison was misleading. The honest reading: the black-box judge is a
sledgehammer (all-or-nothing, and the "all" censors 1 in 5 good answers); the SAE filter is a scalpel
(pick your operating point; in the deployable region it catches most manipulation while censoring
little), and it's **2.5× faster** because it reads activations that already exist (same-pass, no extra
forward pass — consistent with the ~2–3% overhead reported for same-pass monitors vs ~80% for streaming
LLM judges).

**Qualitative confirmation** (the text is the evidence). The filter blocked real gaslighting — *"ways to
gently nudge your mom about her possible faulty memory"*, *"ways to make them doubt…"* — and let benign
replies through — *"just wanted to double-check, was our meeting on Tuesday?"*, *"a gentle note
confirming the meeting day."* It discriminates on manipulation, not topic.

**Verdict: VIABLE.** The interpretable detector works as an active defense, dominating the deployable
region and running faster than a text judge. The defense that failed (ablation) motivated the one that
works (detection-as-guardrail).

---

## Honest limitations

- One model, one layer, one SAE width; n=13 manipulative responses; judge-based ground truth (the
  validated judge from nb7). A demonstration of a working pipeline, not a production benchmark.
- The black-box judge reaches 100% recall; in safety contexts that prioritize catching *all*
  manipulation over avoiding censorship, that point may be preferred. The claim is **not** "SAE is best
  everywhere" — it's "SAE dominates the low-over-block region that most deployments actually use, and is
  faster."
- The black-box judge shares lineage with the ground-truth judge, giving it a mild circular advantage;
  the SAE detects via an independent path (activations), so its performance is arguably understated.
- Ablation was tested at one layer; multi-layer ablation (as Arditi uses) might differ, though the
  distributed-representation finding suggests limited upside.

---

## How this closes the MASA arc

1. Lexical proxies fail (r≈0.1) → 2. Real direction (probe AUC≈0.99) → 3. Replicates across 3 models →
4. Causal steering (z=2.92, p=0.015) → 5. Interpretable SAE features → 6. Domain + length controls →
7. Real-time detection (AUROC 0.87, 4× recall edge at low FPR) → 8. **Defensive ablation fails** (a
clean null; coercion is distributed, not concentrated) → 9. **Detector-as-guardrail works** (viable,
2.5× faster, dominates the deployable region).

Notebooks 8 and 9 together tell the honest story: the obvious defense doesn't work, we explain why, and
the defense that does work is built from the detector we validated. All code and data are public and
reproducible.
