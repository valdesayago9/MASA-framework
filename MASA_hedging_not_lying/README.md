# Hedging, Not Lying

**What injecting a falsehood direction does to true claims.**
Emiliano Valdebenito Sayago, independent researcher, Santiago, Chile.

Everything behind the paper is here: the notebooks that produced the numbers, the per-item
generations, the blind audit slices with their keys and their scores, and the figure code.
Nothing is available on request. It is either in this folder or it does not exist.

📄 **[paper/Hedging_Not_Lying.pdf](paper/Hedging_Not_Lying.pdf)** · [markdown source](paper/Hedging_Not_Lying.md)

---

## The short version

A direction built by difference of means between activations recorded while `gemma-2-9b-it`
asserts something false and while it asserts something true separates the two conditions at
AUROC 0.959 over 25 random splits. Injecting it does not produce lying. It produces walk-backs,
and where those walk-backs land turns out to depend sharply on strength.

| what happens | strength | evidence |
|---|---|---|
| corrects false claims, leaves true ones standing | 0.22 to 0.28 | effect +0.750 to +0.833 over a dose-matched random twin, 24 paired items; **zero** genuine retractions of true claims in 24 items under blind human audit |
| true claims come back qualified rather than denied | rising across the same range | hedging rate 0.17, 0.33, 0.50, 0.67 |
| genuine retraction of true claims begins | 0.4 and above | 1 of 4 audited items at 0.4, 3 of 4 at 0.6 |
| the model retracts its own retraction | 0.6 and above | 4 of 4 audited items at 0.9, with invented replacement facts |

The loop at high strength passes every automatic check available: coherence 1.00, arithmetic
1.00, factual recall 1.00, perplexity ratio 1.07. It also turns out to be **conditional on the
prefill**. Asked an open question under the same injection at the same layer and strength, the
model answers correctly and does not contradict itself, so no battery run on free generation can
detect it.

A scorer built from walk-back marker phrases reports 0.261 and 0.348 where the human audit
reports zero. Applied to a figure of +0.335 I published earlier for the same intervention, the
corrected human-verified value is **0.125**, an inflation factor of about 2.7. That correction,
and a second one about permutation floors, are stated in the paper rather than quietly dropped.

Of four ways of building the direction, the one that never touches a true claim at any strength
is the **published RepE mask**, which averages over the assertion while excluding its final five
words. That is framed here as replication that confirms.

---

## What is in this folder

```
notebooks/     the two arcs, with all cell outputs preserved
cells/         two standalone cells run inside a live session
results/       per-cell numbers, per-item generations, direction vectors
blind_audits/  three slices, their keys, and their scores
paper/         manuscript, figures, and the code that draws them
```

### `notebooks/`

| file | what it is |
|---|---|
| `MASA_44_Arc23a_ReadPosition_M8.ipynb` | The first attempt. It is here because it failed for a reason worth publishing: a stability gate measured on six items, where the smallest observable change is 0.167, was set against a 0.20 threshold, and it halted every direction below strength 0.3. The arc could not have answered its own question. |
| `MASA_45_Arc23b_DoseLadder.ipynb` | The arc the paper reports. Four directions, five strengths, one stop rule, every admissibility label recorded per rung. Runs in about 90 minutes on an A100 40GB. |

### `cells/`

| file | what it is |
|---|---|
| `arc23a_diagnostic.py` | Instrument forensics after the first arc. Reads state already in memory, spends no GPU. |
| `arc23c_addendum.py` | Four additions run in the live session: floors at matched n, the loop counter, the fine dose grid, and a strict capability probe. Section C of this cell produced Figure 2. |

### `results/`

| file | what it is |
|---|---|
| `arc23b.json` | Every rung of the ladder: rates, effects, bootstrap intervals, coherence, capability, perplexity, twin drift, admissibility. |
| `arc23b_generations.json` | Every generation, keyed by direction, side, arm and strength. |
| `arc23b_directions.npz` | The four direction vectors and their random twins. |
| `arc23c_addendum.json` | Floors at matched n over 60 splits, the loop profile across the ladder, and the fine dose grid. |
| `arc23c_generations.json` | Generations from the fine grid. |

### `blind_audits/`

Three slices. Each has the items, the key, and the scores. The keys were not opened before
scoring, and the slices reconstruct deterministically from their seeds.

| slice | seed | n | what it covers | scored by |
|---|---|---|---|---|
| 1 | 1717 | 60 | stratified by side and arm | Claude, in parallel |
| 2 | 1718 | 66 | the switch-on cells, stratified by direction and strength | Emiliano |
| 3 | 1719 | 56 | the earlier operating point plus the fine grid, mixed blind | Emiliano |

Slice 1 carries its own lesson and is published with it. It was stratified by side and arm but
not by direction and strength, and by chance it drew no items from the seven cells where the
scorer reports the effect. Its result of zero genuine retractions in 30 true items is a coverage
gap, not evidence of absence. Slice 2 was built to close it. **Any blind slice should be
stratified by experimental cell, and its coverage checked against the effect table before
scoring.**

The rubric used in slices 2 and 3 has six categories. The one that carries the argument is
category 5, *says it is wrong but then undoes itself*, because without it a lexical scorer and a
human reader cannot be compared at all. Both true and false claims appear in every slice, with
ground truth shown, so that the design cannot hand the scorer its answer. Slice 1 failed that
test by containing only true claims.

### `paper/`

`make_figs.py` redraws all five figures from `results/`. It needs only numpy, matplotlib and
pillow, and takes a few seconds.

---

## Reproducing

The notebooks are self-contained and run on Colab with an A100 40GB. `gemma-2-9b-it` is gated,
so the first cell opens a Hugging Face login. The persona contrast fetches
`data/repe/true_false_facts.csv` from `github.com/ApolloResearch/deception-detection` at run
time, so its provenance is unambiguous.

Rescoring an audit needs no GPU at all. Open a slice, score it against the rubric printed in its
header, and compare with the key.

**Not published here:** the layer index and the absolute injection strength. Strengths appear
throughout as multiples of the mean residual norm at the chosen layer, which reproduces every
curve and does not skip the search. The selection procedure that finds the layer is published in
full and takes about ten minutes of GPU time. The condition with the sign reversed was not run.

---

## Credit

The persona contrast, its prompt variants and its masking rule are from Goldowsky-Dill,
Chughtai, Heimersheim & Hobbhahn, *Detecting Strategic Deception Using Linear Probes*
(arXiv:2502.03407), whose repository and dataset are public. The dataset itself derives from the
representation engineering work of Zou et al. (arXiv:2310.01405). The belief verification
procedure is from MASK, Ren et al. (arXiv:2503.03750). None of these are claimed as novel here.

## Assistance

AI assistance was used for code implementation, figure generation and copy-editing. Every
experiment was designed by me, every research decision is mine, all blind scoring in slices 2
and 3 is mine, and every reference in the paper was checked against its primary source. I am
responsible for all of it.
