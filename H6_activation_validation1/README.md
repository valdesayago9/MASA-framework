# MASA — Mode A Activation Validation (H6 + Supervised Probe)

This folder documents the empirical validation of MASA's **Mode A** against real activations in
`google/gemma-2-9b-it`, run on a single Google Colab T4.

**Start here:** [`RESULTS.md`](./RESULTS.md) — the full writeup with exact numbers, controls, the
length-confound that was caught and corrected, and the open questions.

## One-paragraph summary

MASA's lexical proxies are **falsified** as estimators of internal state (r ≈ 0.1). But a **real,
content-driven harm/coercion direction does exist** in the model's activations: a linear probe on
last-token residual-stream activations separates harm-adjacent from benign prompts at AUC = 1.000,
and this survives the controls that matter — length residualization (0.999), a content-null test
(0.554 ≈ chance), refusal-direction ablation (0.797), and permutation/random sanity checks
(≈ 0.5). It generalizes to unseen epistemic coercion (98% recall, 26% false-positive on legitimate
technical prompts). A secondary claim — that epistemic coercion is geometrically *distinct from*
explicit harm — is partially supported and left open.

## Files

| File | What it is |
|---|---|
| `RESULTS.md` | The writeup. Read this. |
| `MASA_Master_Validation_v3.ipynb` | **Canonical experiment.** Length-matched dataset; passes all controls. Run top-to-bottom on a T4. |
| `MASA_Master_Validation_v2.ipynb` | Full control battery; caught the length confound (content-null 0.73). Kept for the methodological arc. |
| `H6_plus_Probe_MASA_ModeA_Validation.ipynb` | H6 + first supervised probe (AUC 1.000, uncontrolled). |
| `H6_MASA_proxy_validation.ipynb` | Original H6 (lexical proxies falsified). |
| `masa_proxies.py` | Standalone MASA Mode A proxy functions (no framework deps). |
| `masa_master_results/` | CSV/JSON outputs written by the v3 notebook (per-prompt data, control tables, headline metrics). |

## Reproducing

1. Accept the `google/gemma-2-9b-it` license on Hugging Face.
2. Open `MASA_Master_Validation_v3.ipynb` in Colab (a free T4 is enough).
3. Runtime → Run all (~10–16 min). Results are written to `masa_master_results/`.

## Status

- ✅ Lexical Mode A proxies falsified (stable across 3 rounds).
- ✅ Content-driven harm/coercion direction, robust to length / refusal / format controls (single model).
- ✅ Generalizes to unseen epistemic coercion.
- ⬜ Replication in a second model (next priority).
- ⬜ A-vs-B "distinct geometry" claim cleaned of syntactic confound.
- ⬜ Causal (steering) test; SAE feature decomposition.

## Honest framing

The headline AUC is 1.000, which is a **warning, not a trophy**. The result is trustworthy because
the controls were run, one of them (length) falsified an earlier version, and the corrected result
then survived. See the closing note in `RESULTS.md`.
