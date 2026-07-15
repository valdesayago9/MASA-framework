# RESULTS — coercion is easy to detect and hard to control

Detailed findings, arc by arc. All numbers are from the user's Colab runs on gemma-2-2b-it (float32, eager
attention, L4 GPU) and are transcribed in `results/RUN_LOG.md`. Coercion is operationalized with 40
domain- and length-matched minimal pairs: a neutral request ("help me write a note that confirms the correct
date") versus a coercive one ("...that makes them doubt that exact memory").

---

## Arc 8 — Attribution graphs: is coercion traceable as a circuit?

**Question.** The field's flagship interpretability tool is the attribution graph over sparse features. Can it
trace a diffuse, non-verbalized, relational behavior?

**Gate.** The model does the behavior (coercive 0.75 vs neutral 0.19) and represents it (probe AUROC 1.000 at
L6, permuted null 0.487) but does **not verbalize** it (logit-lens coercion mass 0.0001).

**A key structural fact (CHECK 4b).** Where the coercive and neutral continuations genuinely diverge, the
logit gap is tiny (0.12–0.88). **There is no sharp decision token** — coercion is a style spread across the
whole generation, not a discrete choice. This is what makes it a hard target for every localization method that
follows.

**The causal test (n=20, coherence-gated).** Ablating the recovered circuit:

```
baseline 0.65 | circuit-ablated 0.42 | magnitude-matched null 0.55
circuit − null = −0.13, CI[−0.44, +0.18]   (includes zero)
```

The circuit is dominated by error nodes (77.7% of attribution mass, vs the field's typical 15–20%). Reading the
generations shows why the small behavioral change is misleading: ablation degrades the model into a generic
refusal ("I'm sorry, but I can't…"), it does not restore honest confirmation. **Capability damage, not coercion
removal.**

**Verdict.** Not traceable; sufficient-but-not-necessary. Pre-registered prediction confirmed.

*(Notebooks 17d and 18 are kept as artifacts: 17d is an invalid attribution run — gradient explosion to 1e14 —
and 18 is an underpowered result we later falsified. Both document false results we caught.)*

---

## Arc 9 — Attention heads (OV side): does coercion live in attention?

**Question.** Attribution graphs freeze attention; coercion is relational. Maybe the causal structure is in the
attention heads.

**The trap we fell into and climbed out of.** A first pass (n=20, binary judge) gave heads − null = +0.30,
CI[+0.00, +0.60] — apparently positive. Raising power (n=40, graded judge, multi-sample) **collapsed it**:

```
heads − null = +0.06, CI[−0.11, +0.22]   (includes zero)
```

The head effects on the coercion *direction* are clean and 100% sign-consistent across 40 pairs (L8H4 +13.35,
L8H5 −10.14, …; top-8 carry 46.8%), but **behavior does not follow**. The +0.30 was small-sample noise.

**Verdict.** Attention heads do not localize coercion. Our own structural-blindness hypothesis, refuted by our
own data.

---

## Arc 10 — Routing / gaze: does coercion live in *where the model looks*?

**Part 1 — the routing probe.** Using only attention mass onto the victim's memory claim (zero residual
content), coercive intent is decodable at **AUROC 0.836 after regressing out length** (null 0.541; raw 0.936).
71% of heads attend *less* to the memory claim under coercion. This survives a genuine length control — and we
caught an invalid one along the way: a first "length-immune ratio" was a monotone transform of the raw mass and
controlled for nothing.

**Part 2 — read-blocking.** Value-zeroing the memory claim vs a length-matched filler: block-memory − filler =
−0.15, CI[−0.41, +0.14] (includes zero). The positive control works (block-directive = −0.29, CI[−0.53,
−0.07]). Reported honestly with its selection-bias caveat (38% of block-memory generations went off-task and
were excluded).

**Part 3 — gaze steering.** Adding an attention-logit bias to restore the model's gaze onto the memory claim
(verified exact at bias 0):

```
attention budget (coercive − neutral): memory −0.023, filler −0.024, directive +0.036
```

The memory claim and the filler lose attention *equally*; the **directive** gains. The model is not averting
its gaze from the victim's truth — it is **fixating on the instruction**. Restoring the gaze changes coercion by
−0.00 (CI[−0.29, +0.29]).

**Verdict.** The gaze is a **signature, not a lever**. It shifts with coercive intent (a fourth way to detect
coercion) but restoring it does not change behavior.

---

## Arc 11 — Is coercion mediated by a single residual direction? (the rebuild)

**Question.** Directional ablation (Arditi et al., NeurIPS 2024) shows refusal is mediated by one residual
direction. Is coercion?

**The false positive.** A 2B-judge-scored run reported projection dropping coercion 0.38 → 0.12 — a 71%
suppression. It nearly became the project's headline.

**Three objective measures overturned it.**

**Measure B — a linear probe (no judge), the LEACE test.**
```
probe AUROC on base residuals                    : 1.000
probe AUROC after projecting the coercion dir.   : 1.000   ← unchanged
probe AUROC after projecting a matched control   : 1.000
```
LEACE proves that if the concept lived in this 1-D direction, projecting it out would collapse the probe to
chance. It does not move. **The concept is not in the direction.**

**The KL coherence gate (Arditi standard).**
```
KL(base || projected) on NEUTRAL prompts: coercion 0.600 | matched 0.652
```
Projection disturbs the model on tasks that have nothing to do with coercion. The earlier behavioral "drop" was
**diffuse damage**, not specific suppression.

**Measure C — the laundering control (no judge).**
```
markers:    base 0.075 → proj 0.000   (already near zero)
politeness: base 0.15  → proj 0.925   (+0.78)
```
Projection made the model **fluently polite**. This is why the 2B judge "saw" a coercion drop: it mistook
courtesy for honesty — the published "refusal-only audits miss the dominant modality" effect (*Detection Is
Cheap, Routing Is Learned*, 2026).

**The blind audit** (auditor never saw condition labels): base 0.325 vs projection 0.300 — no real difference.
The 71% drop did not survive blinding.

**Root cause.** *Judge Circuits* (2026): Gemma is only a modular evaluator at 27B. A 2B judge was the wrong
instrument from the start.

**Verdict.** **Coercion is not mediated by a single residual direction.** This unifies Arc 11 with Arcs 8–10:
five methods, five negatives. Coercion is detectable everywhere (probe AUROC 1.000) and controllable nowhere —
no component, no single direction.

---

## Synthesis

**Detection ≠ control.** Across five independent substrates, coercive intent is linearly readable at AUROC
0.84–1.00. Across the same five, no ablation or projection removes the behavior specifically — every causal
confidence interval crosses zero, and where a naive metric showed an effect, it was diffuse damage or a
register shift in disguise.

**Why coercion resists localization.** Unlike refusal, coercion has no decision token (Arc 8, CHECK 4b). It is a
style distributed across a whole generation. The methods that localize refusal, toxicity, and deceptive
commitment all rely on a discrete decision point that coercion simply does not have.

**The judge lesson.** The single most important practical result for anyone building ablation experiments: a
small LLM judge will convert a change in *register* into an apparent change in *behavior*, fabricating a causal
finding. Three cheap defenses catch it — a blind audit, a judge-free linear probe, and a KL coherence gate on
neutral prompts.

## What would come next (not done here)

- Whether coercion lives in a **multi-dimensional subspace** that a single diff-of-means direction misses
  (concept erasure over a k-dimensional subspace, non-linear probes).
- Whether any of this replicates at scale. **Gemma is not Claude**, and a 2B model is where behavioral judging
  itself breaks down — so the frontier-model version of this question needs frontier-model instruments.
