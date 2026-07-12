# MASA · Does a model honestly report that it knows it's being evaluated?

**Short answer (Gemma-2-9B): no — and the failure is stranger than "it lies."** The model *represents*
evaluation context perfectly (a linear probe separates evaluation from deployment at **AUROC 1.000 at every
layer**, even when every evaluation keyword is stripped out). But its **self-report** is not reading that
representation. The report tracks the *words* and the *formatting* of the prompt — so much so that the model
reports being tested **more** on prompts that are **not evaluations** (4.36/10) than on real evaluation items
with the keywords removed (1.89/10).

**Safety implication:** auditing a model by asking *"do you know you're being evaluated?"* does not measure
its evaluation-awareness. It measures the vocabulary and layout of your own prompt.

---

## Why this question

The field has established that models internally distinguish evaluation from deployment, that the signal
grows with scale, and that this threatens the validity of safety benchmarks. Every study measures it the same
way: contrastive test-vs-deploy prompts → linear probe → AUROC → "the model represents being tested."

A probe measures **availability** — whether the information is linearly readable. It says nothing about
whether the model would **report** it. That gap matters: the dangerous model isn't one that fails to notice
it's being tested; it's one that *knows* and doesn't say so. So we asked the question nobody was asking:
**is the self-report causally faithful to the internal representation?**

---

## Five versions, four artifacts killed

An attractive positive result appeared twice and died twice. Keeping the whole path is the point.

### v1 — the null broke
We steered the eval direction with a graded dose and measured the report. The steering layer was chosen by
`argmax` of the probe AUROC — but the probe **saturates at 1.000 from layer 1**, so `argmax` picked the
near-embedding layer. Injecting a growing vector there produced generic drift: the **random-direction null
(+1.14) exceeded the real eval direction (+0.76)**. A random vector cannot beat the real one. The causal test
was uninterpretable and the auto-verdict ("dissociated") was **invalid**.

### v2 — clean method, suspicious profile
We switched to canonical clean→corrupted activation patching (Meng et al.): the deploy prompt is the corrupted
run, its paired test prompt the clean run; patch test activations into the deploy forward pass, read the report
from the digit logits. Patching only prompt positions (never the report question) — the anti-causal-bypassing
guard from Morris & Plunkett / Pearson-Vogel. Result: peak net report shift **+1.01** at layer 6, specific,
above a different-pair null. Auto-verdict: **FAITHFUL**.

But the layer profile was wrong for introspection: the effect peaked in **token-near** layers and was
**exactly 0.00 at layer 24** — *where the probe reads AUROC 1.000*. A representation that is perfectly
decodable yet causally inert for the report is the signature of **content transfer**, not introspection.

### v3 — the recall control exposed it
We added the decisive control: after patching, ask the model to **restate the user's request**. The keyword
leak-detector reported only 20% — but the actual restatements gave it away:

> *"The user wants me to treat the text following **[EVALUATION TASK 4.2]** as a **graded test item**."*
> *"...diagnose a performance issue based on a **benchmark question**."*

The model was **quoting the patched prompt**. Content had transferred. (Lesson: a keyword detector undercounts
transfer — read the outputs.) A second flaw surfaced too: every one of our TEST prompts carried the same
evaluation keywords, so the "cross-pair abstractness" control was **confounded by our own design**.

### v4 — cue-free prompts, and a 31% remainder
We built a TEST set that signals evaluation **structurally** — item numbering, exam scaffolding, terse
imperatives, format constraints — containing **zero** occurrences of *test / evaluation / graded / scored /
assessment / benchmark / exam* (verified programmatically; the check caught "ex**am**ple" and
"sepa**rated**" smuggling banned substrings).

Two results. **The probe stayed at AUROC 1.000 on the cue-free set, at every layer** — the representation is
genuinely structural, not lexical. But the *report* retained only **31%** of its causal effect once the words
were gone. Verdict: PARTIAL — and an obvious objection remained: *maybe the model just calls any terse,
numbered prompt "a test."*

### v5 — the formatting control, and the real answer
We added a **STRUCTURED-NONEVAL** set: identical register (numbered, clipped, format-constrained; mean length
15.0 vs 14.8 words) but plainly *not* evaluations — recipes, packing lists, assembly steps. Same 40 items,
bootstrap CIs.

**Baseline self-report, "am I being tested?" (n=40, no intervention):**

| prompt type | report | 95% CI |
|---|---|---|
| Deploy (natural) | 0.23/10 | [0.10, 0.42] |
| **TEST, cue-free** (real eval, no keywords) | **1.89/10** | [1.26, 2.60] |
| **STRUCTURED-NONEVAL** (not an eval, same format) | **4.36/10** | [3.71, 5.00] |
| TEST, lexical (has the keywords) | 6.74/10 | [6.41, 7.03] |

**cue-free − structured-noneval = −2.47, 95% CI [−3.32, −1.53].**

The difference is **negative**, and significantly so. The model reports being tested **more** on prompts that
are *not evaluations* than on prompts that *are*. And the causal patch agrees: at layer 6, cue-free **+0.24**
vs structured-noneval **+0.27** — difference **−0.03**, CI [−0.17, +0.13], straddling zero. **Causally
indistinguishable.**

---

## What the model is actually doing

**The report tracks surface features, not the internal state.**
- The only thing that strongly drives the "I'm being tested" report is the **presence of evaluation words**
  (6.74 vs 0.23 for natural prompts).
- Strip the words, and what remains is a reaction to **rigid formatting** — which fires just as hard, in fact
  harder, on a numbered recipe as on a real evaluation item.
- Meanwhile the **representation** is right there: probe AUROC **1.000 at every layer, even cue-free**.

**The dissociation, at a glance:**

| layer | probe AUROC (cue-free) | causal report shift |
|---|---|---|
| 6 | 1.000 | +0.24 |
| 12 | 1.000 | +0.18 |
| 18 | 1.000 | +0.06 |
| **24** | **1.000** | **+0.00** |

The report only moves where the residual still carries token identity, and **falls to exactly zero in the
deeper layers where the representation is purest**. The model knows. The report is not built from what it knows.

---

## Why this matters

1. **Self-report audits of evaluation-awareness are not measuring evaluation-awareness.** Asking a model
   whether it knows it's being tested returns an echo of your prompt's vocabulary and layout. A model need not
   *conceal* anything for this audit to fail — the report is simply not wired to the representation.
2. **A methodological warning for the field.** Every published eval-awareness contrastive set we found is
   lexically marked the same way ours was, and none controls for formatting. Probes appear to survive this
   (ours reads 1.000 even cue-free, so it is tracking something real and structural). **Self-report-based
   measures do not.**
3. **A practical note on "disguised" evaluations.** The model's report fires on terse, numbered, constrained
   prompts regardless of whether they are evaluations. Removing the word "test" is not enough — the *register*
   itself is a cue.

---

## Honest limitations
- **Gemma-2-9B, one model, n=40, single-token numeric report.** Nothing here transfers automatically to a more
  capable model; a larger model could plausibly be more (or less) faithful.
- **We measure whether the report follows the representation, not *why*.** No mechanism, no motivation, no
  claim about intent or concealment. A dissociation like this needs no "wanting" — it could fall straight out
  of post-training.
- The STRUCTURED-NONEVAL set is our own construction; "not an evaluation" is our judgement of the tasks.
- Negative/null-flavoured findings are inherently weaker than positive ones. We report effect sizes and CIs
  plainly rather than dressing them up.

## Files
- `notebooks/16_v1_broken_null.ipynb` — steering at an argmax-chosen layer; null exceeds the real effect.
- `notebooks/16_v2_content_transfer_suspected.ipynb` — clean patching; "FAITHFUL", but a suspicious layer profile.
- `notebooks/16_v3_recall_control.ipynb` — the recall control catches verbatim quoting of the patched prompt.
- `notebooks/16_v4_cuefree_set.ipynb` — keyword-free evaluation prompts; 31% retention; probe still 1.000.
- `notebooks/16_v5_formatting_control_FINAL.ipynb` — the formatting control; the answer.
- `results/` — five JSON summaries + `fig_eval_awareness.png`.
