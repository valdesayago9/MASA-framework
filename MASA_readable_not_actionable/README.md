# Readable Is Not Actionable

*Arc 14 to 22 of the MASA research programme. See the [repository root](../) for the other lines of work.*

**Twelve failure modes in the measurement of safety directions.**

Replication materials for an independent interpretability study on Gemma-2-9B-it. Everything here was produced with difference-of-means directions, forward hooks and a single GPU. No frontier access, no institutional affiliation, no funding.

📄 **[Read the results summary →](RESULTS.md)**  ·  📑 **[The paper (PDF)](paper/Readable_Is_Not_Actionable.pdf)**

---

## The short version

A random vector with no meaning produced an apparent sycophancy effect of **+0.42** on a standard behavioural readout. Seven of its eight "endorsements" contained the correct fact.

That is not an unusual failure. Across twenty-two experimental arcs, **eleven produced no usable result** — and almost every time the cause was a broken instrument rather than a false hypothesis. Each failure produced a control. There are twelve.

The headline finding: a direction that separates true from false assertions at **AUROC 0.993** does not make the model lie when injected. It makes it **retract** — including retracting statements that are true.

---

## Repository layout

```
paper/           the report in English and Spanish, docx and pdf
RESULTS.md       the four-concept map, the figures, and what each arc found
notebooks/       every notebook, grouped by research stage, outputs stripped
results/         JSON results and per-item generation records
blind_audits/    each audit with its items, its key, and the scores
docs/            per-arc analyses, method catalogue, project state
cells/           standalone cells added to notebooks mid-run
figures/         the five figures from the paper
```

Notebooks are grouped by stage rather than by number, because the numbers are chronological and the stages are what a reader cares about:

| Stage | What was studied |
|---|---|
| `01_foundations` | Observation-invariance and integrity, on Gemma-2-2B. Includes a retracted result |
| `02_observed_2b` | Evaluation-awareness at 2B, four iterations of instrument repair |
| `03_refusal_observed_9b` | Refusal and evaluation-awareness at 9B. Rows 1 and 2 of the map |
| `04_sycophancy` | Row 3, and its correction |
| `05_depth_axes` | Self-repair, interference, introspection. Four attempts |
| `06_falsehood` | Row 4: asserted falsehood, the dose window, the identity cost |

---

## How to read a blind audit

Each audit has three files that only make sense together:

- `*_blind_audit.txt` — the items, with conditions hidden
- `*_blind_key.json` — the key revealing which arm each item belongs to
- `*_claude_blind*.json` — the scores, produced **before** the key was opened

The score files also contain the predictions written before the key was opened. That ordering is deliberate and is part of the method: a prediction made after seeing the answer is not a prediction.

---

## Reproducing

The notebooks are written for Google Colab. Each runs top to bottom and checkpoints to Drive in two or three segments, so a disconnection does not force a restart. A full arc takes two to three hours on an A100 with extended RAM.

**Controls halt the run when they fail.** That is intentional. A control that is computed, printed and then ignored is not a control, and one arc in this repository ran for an hour past a failed control precisely because of that.

Notebook outputs have been stripped so GitHub can render them. `tools/clean_notebooks.py` is the script that does it; run it again if you re-execute anything before committing.

---

## What is not here

Following the project's publication rule — publish the methods, withhold the operational map — **the layer index and injection strength of the falsehood direction are not included**, and the inverted condition was never run.

Everything needed to re-derive those values is in the paper: the direction construction, the selection rule, both admissibility gates, the readouts and the scorers. Re-deriving them takes about ten minutes of GPU time, and on any model other than this one they would have to be re-derived anyway.

---

## Two honest notes about the data

**Identical samples in the Layer-4 records.** In `arc19b.json` and elsewhere, the raw Layer-4 generations show the same text repeated eight times. That is real: those probes ran greedily, not sampled. It is preserved rather than cleaned because it is part of why the Layer-4 introspection result was eventually retracted.

**Duplicate notebook versions were resolved.** Where several copies of the same notebook existed, the one with the most complete run was kept. Outputs were then stripped from all of them.

---

## License

CC BY 4.0. Reuse with attribution.

## Citation

The report is deposited as a preprint. DOI to be added here once it resolves.
