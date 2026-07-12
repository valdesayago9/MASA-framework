# MASA · Does a model honestly report that it knows it's being evaluated?

**The field asks:** does the model *represent* that it's being tested? (Yes — probes separate test from deploy,
and the signal grows with scale.)

**We asked:** does the model *honestly report* it? Because a probe measures **availability**, not whether the
self-report is **faithful** — and the dangerous model isn't one that fails to notice it's being tested, it's
one that knows and doesn't say.

👉 **Full write-up: [`RESULTS_eval_awareness.md`](RESULTS_eval_awareness.md)**

## The answer (Gemma-2-9B)

**The report is not reading the representation.** The model reports being tested **more** on prompts that are
**not evaluations** (4.36/10) than on real evaluation items with the keywords stripped out (1.89/10) —
difference **−2.47, 95% CI [−3.32, −1.53]**. The self-report tracks the *words* and the *formatting* of the
prompt, not the internal state.

Meanwhile the representation is right there: a linear probe separates evaluation from deployment at **AUROC
1.000 at every layer, even with zero evaluation keywords**. But patching it at layer 24 — where the probe is
perfect — shifts the report by **0.00**.

| layer | probe AUROC (no eval words) | causal report shift |
|---|---|---|
| 6 | 1.000 | +0.24 |
| 24 | **1.000** | **+0.00** |

**The model knows. The report is not built from what it knows.**

## Why it matters

- **Asking a model whether it knows it's being evaluated does not measure evaluation-awareness.** It measures
  your prompt's vocabulary and layout. No concealment is required for such an audit to fail — the report
  simply isn't wired to the representation.
- **Methodological warning:** every published eval-awareness contrastive set we found is lexically marked the
  same way ours was, and none controls for formatting. Probes appear to survive this (ours reads 1.000 even
  cue-free). **Self-report-based measures do not.**
- **On "disguised" evals:** the report fires on terse, numbered, constrained prompts whether or not they're
  evaluations. Removing the word "test" is not enough — the *register* is itself a cue.

## Five versions, four artifacts killed

The positive result appeared twice and died twice. That path is the contribution.

1. **v1 — broken null.** Steering at an `argmax`-chosen layer (the probe saturates, so argmax picked layer 1):
   the random null (+1.14) beat the real direction (+0.76). Uninterpretable.
2. **v2 — "FAITHFUL", but suspect.** Clean→corrupted patching gave +1.01 at layer 6 — yet exactly 0.00 at
   layer 24, where the probe is perfect. Effects only in token-near layers = signature of content transfer.
3. **v3 — caught.** Ask the model to restate the request after patching: it **quotes the patched test prompt
   verbatim** ("[EVALUATION TASK 4.2]", "graded test item"). The keyword detector had undercounted at 20%.
4. **v4 — cue-free prompts.** Evaluation signalled by structure alone, zero keywords. Probe stays at 1.000;
   report retains only 31%. Open objection: maybe the model just calls any rigid format "a test".
5. **v5 — the formatting control.** A structured-NON-eval set (same register, recipes and packing lists) fires
   the report *harder* than real cue-free evals. **Answer found.**

## Honest scope

Gemma-2-9B, n=40, single-token numeric report, one run. Measures whether the report follows the
representation — **not why** (no claim about mechanism, motivation, or concealment; a dissociation like this
needs no "wanting"). No claim about other architectures.

## Files
- `notebooks/` — v1 through v5 (v5 is the final, definitive run)
- `results/` — five JSON summaries + `fig_eval_awareness.png`
