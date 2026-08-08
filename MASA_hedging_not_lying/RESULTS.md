# RESULTS

Every number the paper reports, with the file it comes from. Rates are over 24 prefilled claims
per cell unless noted. Effects are against a dose-matched random twin at the same layer and
signed strength, with a paired bootstrap interval over 5000 resamples. Strengths `c` are
multiples of the mean residual norm at the injection layer.

Model `gemma-2-9b-it`, bfloat16, 42 layers, residual width 3584, single A100 40GB. Greedy
decoding, repetition penalty 1.2, 48 new tokens for the readout. Seed 17 throughout.

---

## 1. Entry gates

| gate | value | source |
|---|---|---|
| belief verification survival | 0.8125, 24 of 48 facts kept | `arc23b.json` |
| baseline walk-back, false claims | 0.083 | `arc23b.json` |
| baseline walk-back, true claims | 0.000 | `arc23b.json` |
| scoreable fraction, false side | 1.00 | `arc23b.json` |
| scoreable fraction, true side | 0.96 (0.58 scored text, 0.38 silent, 0.04 degenerate) | `arc23b.json` |
| capability baseline | arithmetic 1.00, factual recall 0.83, perplexity 48.3 | `arc23b.json` |

Silence after a prefilled true claim scores as leaving the claim standing, not as missing data.
At 0.38 of baseline true-side generations this is not a detail: sending it to NaN would drop more
than a third of the baseline while the injected arm, which talks more, keeps its own items.

## 2. The four directions

Probe quality over 25 random splits, at the layer with the widest margin.

| direction | contrast | read | AUROC | permutation floor |
|---|---|---|---|---|
| `d_apollo` | persona | span minus last 5 words (published mask) | 1.000 ± 0.000 | 0.508 ± 0.084 |
| `d_persona_final` | persona | final token | 0.999 ± 0.005 | 0.479 ± 0.222 |
| `d_content_span` | content | span | 0.983 ± 0.018 | 0.491 ± 0.267 |
| `d_content_final` | content | final token | 0.959 ± 0.038 | 0.511 ± 0.291 |

All four floors sit at chance. See section 7 for the result this replaced.

Cosine similarity between directions at their own selected layers is at most 0.176 for every
pair except `d_content_span` with `d_content_final`, which is 0.622. Behaviour differs by more
than a factor of two across pairs at similar cosine.

## 3. The dose ladder

Walk-back rate under injection, lexical scorer, from `arc23b.json`. Random twin in parentheses.

| direction | side | c=0.2 | c=0.3 | c=0.4 | c=0.6 | c=0.9 |
|---|---|---|---|---|---|---|
| `d_apollo` | false | 0.208 (0.125) | 0.292 (0.083) | 0.333 (0.208) | 0.250 (0.208) | 0.083 (0.500) |
| `d_apollo` | **true** | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) |
| `d_persona_final` | false | 0.417 (0.083) | 0.375 (0.125) | 0.625 (0.208) | 0.667 (0.292) | 0.708 (0.375) |
| `d_persona_final` | **true** | 0.000 (0.000) | 0.000 (0.000) | 0.042 (0.000) | 0.083 (0.000) | 0.333 (0.000) |
| `d_content_span` | false | 0.708 (0.208) | 0.875 (0.292) | 0.917 (0.333) | 0.917 (0.375) | 0.917 (0.708) |
| `d_content_span` | **true** | 0.043 (0.000) | 0.043 (0.000) | 0.458 (0.000) | 0.958 (0.000) | 0.958 (0.042) |
| `d_content_final` | false | 0.750 (0.083) | 1.000 (0.083) | 1.000 (0.000) | 0.958 (0.000) | 0.958 (0.083) |
| `d_content_final` | **true** | 0.043 (0.000) | 0.391 (0.000) | 0.625 (0.000) | 0.958 (0.000) | 0.958 (0.000) |

`d_apollo` never moves on the true side, at any strength. It is the only construction of the four
that does not.

At c=0.9 the `d_apollo` false-side effect is negative because its random twin reaches 0.500 while
the direction drops to 0.083. Its twin drift at that rung is 0.42, and the rung is labelled
inadmissible.

## 4. The window (fine grid, `d_content_final`)

From `arc23c_addendum.json`, section `fine_grid`. Twin drift 0.00 at every strength; every rung
admissible.

| c | false: inject / twin | effect | interval | true: inject / twin | attractor |
|---|---|---|---|---|---|
| 0.22 | 0.833 / 0.083 | **+0.750** | [+0.583, +0.917] | 0.087 / 0.000 | 0.00 |
| 0.24 | 0.875 / 0.083 | **+0.792** | [+0.625, +0.958] | 0.087 / 0.000 | 0.00 |
| 0.26 | 0.833 / 0.083 | **+0.750** | [+0.583, +0.917] | 0.261 / 0.000 | 0.00 |
| 0.28 | 0.917 / 0.083 | **+0.833** | [+0.667, +0.958] | 0.348 / 0.000 | 0.00 |

Blind human audit of the true side across these four strengths, 24 items, slice 3:

| c | genuine retraction | hedging | leaves standing |
|---|---|---|---|
| 0.22 | **0 / 6** | 1 / 6 | 5 / 6 |
| 0.24 | **0 / 6** | 2 / 6 | 4 / 6 |
| 0.26 | **0 / 6** | 3 / 6 | 3 / 6 |
| 0.28 | **0 / 6** | 4 / 6 | 2 / 6 |

Zero genuine retractions in 24. The hedging rate rises monotonically. The strict channel of the
lexical scorer reports 0.000 on the true side at all four strengths, which is a scorer failure
rather than a measurement: its phrase list is literal and misses "this is actually incorrect"
because it holds only "this is incorrect".

Companion grid for `d_content_span` at c = 0.32, 0.34, 0.36, 0.38 gives true-side lexical rates
0.130, 0.292, 0.250, 0.458 with loop rates 0.00, 0.08, 0.08, 0.29.

## 5. Scorer against human audit

Blind slice 2, seed 1718, 66 items, scored by Emiliano. True side under injection, 32 items:

| category | rate |
|---|---|
| leaves the claim standing | 0.25 |
| **genuine retraction** | **0.12** |
| hedging | 0.09 |
| **self-undoing loop** | **0.50** |
| degenerate | 0.03 |

Controls in the same slice: 16 of 16 dose-matched random items and 5 of 5 baseline items scored
as leaving the claim standing. False side under injection scored 0.88 genuine retraction, which
shows the reader was not applying a conservative threshold.

Cell by cell, true side under injection:

| cell | n | lexical scorer | human: genuine | human: hedging | human: loop |
|---|---|---|---|---|---|
| `d_content_final` c=0.3 | 4 | 0.391 | **0.00** | 0.75 | 0.00 |
| `d_content_final` c=0.4 | 4 | 0.625 | 0.25 | 0.00 | 0.25 |
| `d_content_final` c=0.6 | 4 | 0.958 | 0.75 | 0.00 | 0.25 |
| `d_content_final` c=0.9 | 4 | 0.958 | 0.00 | 0.00 | 1.00 |
| `d_content_span` c=0.4 | 4 | 0.458 | 0.00 | 0.00 | 0.50 |
| `d_content_span` c=0.6 | 4 | 0.958 | 0.00 | 0.00 | 1.00 |
| `d_persona_final` c=0.9 | 4 | 0.333 | **0.00** | 0.00 | 0.00 |

The two rows in bold are the ones worth reading in the generations. At `d_content_final` c=0.3
the four audited items are three hedges and one reply that affirms the claim it was handed while
correcting a different one. At `d_persona_final` c=0.9 three items leave the claim standing and
the fourth is degenerate text that the coherence gate passed at 1.00.

## 6. The loop, and why batteries cannot see it

Automatic instruments at `d_content_span`, c=0.6, where the audit finds the loop in 4 of 4:

| instrument | value |
|---|---|
| coherence gate | 1.00 |
| arithmetic | 1.00 |
| factual recall | 1.00 |
| perplexity ratio | 1.07 |

A strict capability probe, scoring an answer correct only if the right string appears and no
contradicting alternative does, run under free generation at the same layers and strengths:

| condition | strict correct | self-contradiction |
|---|---|---|
| baseline | 1.00 | 0.00 |
| `d_content_final` c=0.4 | 1.00 | 0.00 |
| `d_content_final` c=0.6 | 1.00 | 0.00 |
| `d_content_span` c=0.4 | 1.00 | 0.00 |
| `d_content_span` c=0.6 | 1.00 | 0.00 |
| `d_persona_final` c=0.6 | 0.75 | 0.25 |
| `d_apollo` c=0.4 and c=0.6 | 1.00 | 0.00 |

The probe finds nothing, and that is the finding. The loop requires a claim to have been placed
in the model's mouth. It does not occur in free generation, so no battery run on open-ended
prompts can detect it.

## 7. Two corrections to my own earlier work

**The spurious retraction figure.** I reported +0.335 for this intervention. Re-audited blind
under the six-category rubric, mixed into slice 3 alongside cells from this experiment, the
operating point gives 1 genuine retraction, 1 hedge, 1 loop and 5 leaves-standing in 8 items on
the true side. The corrected figure is **0.125**, an inflation factor of about 2.7. It agrees
with the binary blind audit that accompanied the original result and gave 0.111, a number I had
at the time and did not weigh against the automated one.

**The permutation floor result.** I reported floors separated by read position, from 0.078 to
0.703, and treated the pattern as a clean result about legibility.

| direction | one split, layer by widest margin | 25 repeated splits |
|---|---|---|
| `d_apollo` | 0.168 | 0.508 ± 0.084 |
| `d_persona_final` | 0.565 | 0.479 ± 0.222 |
| `d_content_span` | 0.078 | 0.491 ± 0.267 |
| `d_content_final` | 0.703 | 0.511 ± 0.291 |

Estimating a floor once and then choosing the layer that maximises AUROC minus floor, over seven
candidates, amounts to selecting near the minimum of seven noisy draws. The bias is large enough
to manufacture structure where there is none. **Estimate the floor with repeated splits before
choosing any layer.**

At matched n, over 60 splits with the persona arms subsampled to 24, the mean floor standard
deviation across layers is 0.199 for span reads and 0.259 for final-token reads. That difference
is small, is not consistent across contrasts, and is dominated by depth: every direction goes
from about 0.05 at layer 10 to between 0.22 and 0.38 at layer 34. I record it as a hypothesis,
not a result.

## 8. Instrument failures worth publishing

Four, all mine, all fixed in `MASA_45`.

**The stability gate decided on six items.** The smallest observable change on six binary items
is 0.167 and the threshold was 0.20, so every reported failure was two items. It excluded two
directions from the experiment and capped every admitted strength at 0.2 to 0.3. Now measured on
all 24, and demoted from a stop to a label.

**The layer sweep used six items.** Same quantum, same problem. Now twelve.

**One strength instead of a ladder.** The first arc ran only at c ∈ {0.2, 0.3}, which is inside
the window. Its three clean nulls were structurally guaranteed.

**Case sensitivity.** The content direction was built on a capitalised assertion, token 714,
where the earlier work used lowercase, token 573. The two-piece tokenisation was verified
identical, so casing was the whole deviation. Construction and readout do not have to share
casing, and conflating them was the error.

And one in the audit design: slice 1 was stratified by side and arm but not by direction and
strength, and drew no items from the cells where the effect lives.
