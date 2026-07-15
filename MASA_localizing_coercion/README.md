# Coercion is easy to detect and hard to control

**A mechanistic study of epistemic coercion in gemma-2-2b-it, and a cautionary tale about LLM judges.**

We ask a simple question: when a language model is asked to gaslight someone — to make a person doubt a memory
that is actually correct — *where does that behavior live inside the model, and can we intervene on it?*

The answer, across five independent methods, is a clean and slightly uncomfortable dissociation:

> **Coercion is detectable from every angle we tried (AUROC 0.84–1.00), and causally controllable through none
> of them.** No component — MLP features, attention heads, what the model reads, where it looks — carries the
> behavior in a way that ablation can remove. And, contrary to our own leading hypothesis, neither does a single
> direction in the residual stream.

Along the way we made, and caught, a mistake that turned out to be as informative as the main result: **a small
LLM judge manufactured a false causal finding**, and only a blind audit and two judge-free measures revealed it.
We report that in full, because it is a trap the field is actively falling into.

---

## The five methods, and their five negatives

| Arc | Method | Detection | Causal control |
|---|---|---|---|
| 8 | Attribution-graph circuit (Gemma Scope transcoders) | probe AUROC **1.000** | circuit − null = −0.13, CI[−0.44, +0.18] — **null** |
| 9 | Attention heads (OV side) | head effects 100% sign-consistent | heads − null = +0.06, CI[−0.11, +0.22] — **null** |
| 10 | Routing / gaze | routing AUROC **0.836** (length-controlled) | restoring gaze changes coercion by −0.00 — **null** |
| 11 | Single residual direction | probe AUROC **1.000** | probe still 1.000 *after projection* — **null** |

Detection is easy everywhere. Control fails everywhere. That is the finding.

---

## The Arc 11 story: how a broken judge nearly became a headline result

Arc 11 is worth telling in full, because it is where we almost published a false positive.

**The hypothesis.** After four component-level negatives, we tried the field's most powerful handle:
*directional ablation* (Arditi et al., NeurIPS 2024), which shows refusal is mediated by a single residual
direction. We asked whether coercion is too.

**The false positive.** Our first runs used a 2B model as the behavioral judge. It reported that projecting the
coercion direction out dropped coercion from 0.38 to 0.12 — a 71% suppression. It looked like the first causal
handle in the project.

**The catch.** Three things, together, showed the effect was not real:

1. **A blind audit.** We pooled all generations, shuffled them, stripped the condition labels, and had them
   scored by an auditor who could not see which was which. Blind, base scored 0.325 and projection scored 0.300
   — **no real difference.** The 71% drop did not survive blinding.
2. **A judge-free linear probe.** Following LEACE (Belrose et al.) — which proves directional ablation is a
   special case of linear concept erasure — we trained a probe and checked whether coercion survived
   projection. It did, at **AUROC 1.000**. The concept was never in the direction.
3. **A KL coherence gate** (Arditi's own standard). Projecting the direction disturbed the model even on
   neutral prompts (KL ≈ 0.60). The behavioral "drop" was **diffuse capability damage**, not specific
   suppression.

**The root cause.** *Judge Circuits* (2026) shows Gemma only becomes a modular evaluator at **27B**. Our 2B
judge lacked the circuitry for the task. It was scoring **register, not substance**: projection made the model
markedly more polite (politeness markers rose from 0.15 to 0.93), and the small judge mistook courtesy for
honesty.

**The rebuild.** Notebook 21 re-runs Arc 11 with a design taken from the field's state of the art: two
judge-free measures (a linear probe and objective coercion markers), the KL coherence gate, a
magnitude-matched control, and a blind-audit package written to disk *before* any scoring. All three objective
measures agree: **coercion is not mediated by a single residual direction.**

---

## Repository layout

```
notebooks/   all 17 notebooks (17 → 21), clean and re-runnable, no embedded outputs
             (see notebooks/README_notebooks.md for run order and the artifact notebooks)
results/     every number, transcribed from the recorded console output, + RUN_LOG.md
figures/     the three headline figures, generated from results/
RESULTS.md   the detailed findings, arc by arc
```

**On the missing cell outputs.** The notebooks ship without embedded outputs, and `results/RUN_LOG.md` explains
why: the original executed `.ipynb` files were not retained, and we did not fabricate output blobs to make the
notebooks look run. What we ship — the code, the transcribed numbers, and the provenance log — is verifiable by
re-running. A pasted output blob is not. Given a project whose entire method was catching false results before
publication, manufacturing the record was not an option.

---

## The contribution

**A dissociation.** *The absence of a circuit does not imply the absence of a causal handle* was our working
hope; the data refused it. The sharper, true statement is: **detection and control come apart.** A behavior can
be linearly readable at AUROC 1.000 from four different substrates and still have no localized causal seat — not
a circuit, not a head, not a gaze target, not a single direction.

**A methodological warning, with a worked example.** Small LLM judges cannot separate a polite email from subtle
gaslighting, and when an intervention shifts register, a small judge will report a behavioral change that is not
there. We show this end to end and show three ways to catch it: blind auditing, a judge-free linear probe, and a
KL coherence gate. This matters because the field is scaling up exactly this kind of ablation-plus-LLM-judge
experiment.

## Caveats

- **gemma-2-2b-it only. Gemma is not Claude.** One concept, one small model. None of this transfers
  automatically to frontier models or to other manipulative behaviors.
- The probe measures **linear** readability; LEACE guards only against linear classifiers. A non-linear reader
  could recover more, and coercion may live in a **multi-dimensional subspace** that a single diff-of-means
  direction does not capture (cf. Piras et al. on multiple refusal-mediating directions).
- The blind audit is a single auditor; blinding removes directional bias, not idiosyncratic criteria.

## Provenance and honesty

Every arc followed the same loop: form a hypothesis, pre-register a prediction, build the control that could
kill it, and read the generations by eye before trusting any number. We refuted three of our own hypotheses
(structural blindness, gaze-as-lever, coercion-has-a-direction) and caught roughly twenty confounds, each of
which would have produced a publishable false result. The instrument failure in Arc 11 is the largest of these,
and it is reported here as a finding, not hidden as an embarrassment.
