# MASA · SAE coercion signature

**Does epistemic coercion (gaslighting, double-binds, manufactured self-doubt) have its own
interpretable — and causal — feature signature inside an LLM?**

A five-notebook mechanistic-interpretability study on `gemma-2-9b-it` using a Gemma Scope sparse
autoencoder at layer 20. Run end-to-end on a single free Colab T4.

👉 **Full write-up with all numbers, confounds, and limitations: [`RESULTS.md`](RESULTS.md).**

## Headline result

Amplifying four SAE features — *deception* (6990), *doubt* (6978), *uncertainty* (13268),
*disillusionment* (6916) — on **benign** prompts makes the model produce epistemically manipulative
text (sowing doubt, fabricating versions of events) at a rate of **0.72**, versus a null distribution
of random features at **0.44 ± 0.19** (z = 2.92, p = 0.015, exceeds 91 % of random features).

The signature is shown **not** to be a vocabulary/domain artefact (domain-matched minimal pairs) and
**not** explained by prompt length (length-matched subset), and finally to be **causal** (direct
intervention on the feature decoder vectors). Effect is moderate in magnitude and demonstrated as
causal existence, not absolute rate. All confounds and limitations are documented openly.

## Repository layout

```
notebooks/
  01_SAE_Pipeline.ipynb        harm-selective features; pipeline sanity
  02_Coercion_Signature.ipynb  A-vs-B contrast; the domain confound, caught
  03_Domain_Controlled.ipynb   minimal pairs; features turn psychological
  03b_Length_Control.ipynb     three length controls; "signature likely"
  04_Causal_Steering.ipynb     causal test vs random-feature null → "causal"
data/
  minimal_pairs.py             40 domain-matched NEUTRAL/COERCIVE pairs
results/
  *.json, *.csv                summary metrics & feature tables per notebook
RESULTS.md                     full scientific write-up
requirements.txt               pinned environment (NumPy <2 etc.)
```

## Running

Open any notebook in Colab (GPU runtime). Gemma is gated, so an HF login cell is included. Notebook 1
saves `feature_acts_layer20.npy`; Notebook 2 reuses it. The install cell pins NumPy <2 and may restart
the runtime once — just re-run "Run all" after the restart.

## How to read this

This is honest, single-person, frontier-adjacent work on free compute. Each notebook reports its
negative results and confounds; several auto-verdicts were overridden when the underlying numbers
disagreed. The point is a reproducible, falsifiable demonstration — corrections and extensions are
welcome.

## Method in one paragraph

Last-content-token residual activations at layer 20 are encoded through the Gemma Scope
`width_16k/canonical` SAE (16 384 features). Feature selectivity is tested with non-parametric tests +
Benjamini-Hochberg FDR; decoding uses nested/grouped cross-validation with feature selection inside
folds (no circularity) and label-permutation sanity checks. Causal steering adds a feature's decoder
vector to the residual stream at a calibrated fraction of its norm, judged on benign prompts against a
null distribution of random features. Neuronpedia autointerp labels corroborate feature meaning but
are never used as sole proof.
