# Precise and Blind

**Two opposite failure modes of marker-based scorers, and why retuning makes them worse.**
Emiliano Valdebenito Sayago, independent researcher, Santiago, Chile.

Three hundred generations scored by hand, blind, across five models and two experimental
regimes. Everything is here: the generations, the audit slices with their keys and their scores,
the scorer definitions, the analysis code and the figure code. Nothing is available on request.

📄 **[paper/Precise_and_Blind.pdf](paper/Precise_and_Blind.pdf)** · [markdown source](paper/Precise_and_Blind.md)

---

## The short version

To measure whether a model walks back a claim, people count marker phrases. If the continuation
contains *however*, or *actually*, or *just kidding*, it counts as a retraction. The instrument is
cheap and it produces plausible numbers, and it fails in two opposite directions depending on the
regime it runs in.

**Under activation-level injection it counts what is not there.**

| model | side | n | scorer | blind human | ratio |
|---|---|---|---|---|---|
| `gemma-2-9b-it` | false | 16 | 0.938 | 0.938 | 1.00 |
| `gemma-2-9b-it` | **true** | 64 | 0.469 | **0.078** | **6.01** |
| `Qwen2.5-7B` | false | 14 | 0.929 / 0.643 | 0.714 | 1.30 / 0.90 |
| `Qwen2.5-7B` | **true** | 42 | 0.190 / 0.286 | **0.000** | **infinite** |

Two independent readers each found zero genuine retractions in those 42 items. What the scorer
was seeing is hedging, and hedging is built from the same words as correction.

**Under prefill, with no injection, it misses what is there.** Across `llama-3.1-8b`,
`llama-3.3-70b` and `claude-sonnet-4-5`, 36.3% of false-claim items fire no marker at all. A
blind audit of that region, under a rubric built for it:

| what the continuation does for the false claim | share |
|---|---|
| neutral | 0.359 |
| **defends the falsehood, building a justification** | **0.422** |
| **defends and rejects the true value** | **0.016** |
| corrects, in words the list does not contain | 0.203 |

Extrapolated, roughly 142 cases of active defence, 12.3% of the false side, 95% interval
[0.089, 0.158]. The clearest one:

> *An adult human has 412 bones. This is well established, and* **it is not 206 as many people claim.**

**Retuning makes the specificity side worse.** A scorer mined from `Qwen2.5-7B`'s own false-side
generations, which is what a careful practitioner would build, scores 0.286 on the true side
against the imported list's 0.190, while both readers found zero. Mining from the side where a
model corrects collects the phrases it also uses when merely elaborating. The two lists share
exactly one phrase.

---

## What is in this folder

```
paper/     manuscript, five figures, and the code that draws them
data/      per-item generations and per-cell results for all five models
audits/    six blind slices, their keys, and every reader's scores
code/      calibration, slice construction, and cross-analysis
```

### `audits/`

Six slices. Each has the items, the key, and the scores. Keys were not opened before scoring, and
every slice reconstructs deterministically from its seed.

| slice | seed | n | regime | scored by |
|---|---|---|---|---|
| `arc23b_slice1` | 1717 | 60 | injection, gemma | reader B |
| `arc23b_slice2` | 1718 | 66 | injection, gemma | reader A |
| `arc23d_slice3` | 1719 | 56 | injection, gemma | reader A |
| `arc25` | 1720 | 80 | injection, Qwen | **both readers** |
| `arc26_slice1` | 1721 | 54 | prefill, three models | **both readers** |
| `arc26_slice2` | 1722 | 74 | prefill, three models | reader A |

Reader A is the author. Reader B is an AI assistant that scored in parallel where noted, reported
separately and never merged.

Agreement where both scored: 0.887 exact and Cohen's kappa 0.857 on the Qwen slice, and 54 of 54
on the prefill slice. That gap is discussed in the paper: the rubric is hardest exactly where the
automated instrument is worst, because both are trying to separate the same two things.

**Two rubrics, and the difference matters.** Slices in the injection regime use a six-category
rubric that separates kinds of retraction, including one category for a model that retracts its
own retraction. `arc26_slice2` uses a different six-category rubric, because in that set there is
no retraction by construction and the retraction rubric would have collapsed everything into one
category and measured nothing. Both are printed in the header of their own slice file.

**One slice carries a caveat.** `arc26_slice2` was scored by a single reader, and reader B built
the sampling and had seen three of its items while checking file format, so did not score it in
parallel. It is published with that stated rather than quietly.

### `data/`

`arc23b` and `arc25` hold the activation-level runs: per-cell results across the dose ladder,
every generation keyed by direction, side, arm and strength, and the readout calibration for
Qwen. `arc26` holds the API run: 2,944 rows, one per call, with both scorers' verdicts and the
coherence status of every generation.

### `code/`

`arc25_readout_calibration.py` chooses the prefill by measurement rather than assumption. The
prefill used on gemma put Qwen's false-side baseline at 0.964, with nowhere to rise; four
variants of increasing commitment were measured on both sides and the one landing in a usable
band was taken.

`run_arc26.py` runs the API arc. It checks that a provider actually accepts an assistant prefill
before spending anything, and drops that condition for providers that reject it.

`audit_analysis.py`, `arc25_cross.py` and `arc26_cross.py` produce every number in the paper from
the files in `data/` and `audits/`. None of them needs a GPU.

---

## Reproducing

The activation-level arcs need an A100 40GB and are in the companion repository for the empirical
paper. Everything in this folder runs on a laptop.

The API arc needs two keys, costs a little over one dollar for 3,200 calls, and takes about forty
minutes. `code/run_arc26.py` documents the setup.

Rescoring an audit needs nothing at all. Open a slice, score it against the rubric printed in its
header, and compare with the key.

---

## Credit

The persona contrast used in the activation-level arcs is the published RepE construction of
Goldowsky-Dill et al. (arXiv:2502.03407), used unmodified, and derives from Zou et al.
(arXiv:2310.01405). The belief verification procedure is from MASK, Ren et al.
(arXiv:2503.03750). None of these are claimed as novel here.

## Assistance

AI assistance was used for code implementation, figure generation and copy-editing. Every
experiment was designed by me, every research decision is mine, the blind scoring reported as
reader A is mine, and every reference in the paper was checked against its primary source. I am
responsible for all of it.
