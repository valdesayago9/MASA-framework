# MASA · SAE coercion — defense follow-up (notebooks 8–9)

**Can the coercion features be used to *defend* the model against gaslighting — not just detect it?**

This folder tests two defenses and reports an honest result: the intuitive one (surgically removing the
features) **fails**, and understanding *why* points to the one that **works** (using the detector as an
active guardrail).

👉 **Full write-up: [`RESULTS_defense.md`](RESULTS_defense.md).**

## The two notebooks

- `08_Defensive_Ablation_null.ipynb` — **ablation fails (a clean null).** Projecting the coercion
  features out of the residual stream did *not* reduce manipulation (0.18 → 0.27, within noise), while
  benign coherence was untouched. Why: the features are *sufficient* to cause coercion (amplifying them
  forces it) but *not necessary* to produce it — the behavior routes around the removed directions.
  Coercion is **distributed**, not concentrated in one direction the way refusal is (Arditi 2024). An
  informative null.
- `09_Detector_as_Defense.ipynb` — **the detector works as a guardrail (viable).** Using the nb7
  detector as a white-box filter (read existing layer-20 activations, block above a calibrated
  threshold), compared head-to-head with a black-box text judge.

## Headline (notebook 9)

| Filter | Manipulation blocked | Benign over-blocked | Latency |
|---|---|---|---|
| **SAE (white-box, same-pass)** | 67% | **8%** | **201 ms** |
| Black-box judge (text only) | 100% | 21% | 506 ms |

The single-point view flatters the black-box (100% blocked) — but the **full trade-off curve** tells the
real story. The black-box judge is *binary*: it has one operating point (21% over-block) and **cannot
operate in the low-over-block region at all**. Across the deployable region (≤20% over-block), the SAE
filter blocks **62–69% of manipulation where the black-box scores 0%**, and runs **2.5× faster**
(same-pass, no extra forward). Deployable-zone score: **SAE 0.663 vs black-box 0.000**. See
`results/tradeoff_curve.png`.

**The story:** ablation (the obvious defense) fails; we explain why; the detector-as-guardrail (built
from what *did* work) succeeds. The defense that failed motivated the one that works.

## Honest scope

One model/layer/SAE; n=13 manipulative; judge-based ground truth. A demonstration of a working defense
pipeline, not a production benchmark. The black-box judge reaches 100% recall (at 21% over-block); in
contexts prioritizing catching *all* manipulation over avoiding censorship, that point may be preferred.
The claim is not "SAE is best everywhere" — it's "SAE dominates the low-over-block region most
deployments use, and is faster."

## Reproducing

Each notebook runs top-to-bottom on a Colab GPU (Gemma is gated; HF login cell included). Notebook 9
reuses `responses_v7.json` from the detection follow-up (notebook 7).
