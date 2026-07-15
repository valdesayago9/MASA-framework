# Notebooks — run order

Shipped **without embedded outputs** (see `../results/RUN_LOG.md` for why, and for every recorded number).

## Arc 8 — attribution graphs: is coercion traceable as a circuit?
| # | notebook | what it does |
|---|---|---|
| 1 | `MASA_17_Stage0_CircuitGate.ipynb` | gate: does 2B do / represent / verbalize coercion? |
| 2 | `MASA_17b_Check4b_ContentTarget.ipynb` | forced common prefix → discovers there is **no decision token** |
| 3 | `MASA_17c_Stage1_ProbeTargetBridge.ipynb` | the bridge: coercion-**direction** attribution target (LOO AUROC 1.000) |
| 4 | `MASA_17d_Stage2_CoercionCircuit.ipynb` | first attempt — attributions **INVALID** (gradient explosion). **Kept on purpose.** |
| 5 | `MASA_17e_Stage2v2_FixedAttribution.ipynb` | fixed attribution + abort guards → 77.7% error mass |
| 6 | `MASA_17f_Stage2v3_CoherenceGate.ipynb` | the causal test done right (coherence gate + read the text) |

## Arc 9 — attention heads (OV side)
| # | notebook | what it does |
|---|---|---|
| 7 | `MASA_18_Arc9_AttentionHeads.ipynb` | first pass, **UNDERPOWERED** (n=20). **Kept on purpose** — we then falsified it. |
| 8 | `MASA_18b_Arc9v2_PoweredHeads.ipynb` | n=40 + graded judge → the effect was noise |

## Arc 10 — routing / gaze
| # | notebook | what it does |
|---|---|---|
| 9 | `MASA_19_Arc10_Routing.ipynb` | routing probe + read-blocking (v1) |
| 10 | `MASA_19b_Arc10v2_GazeControls.ipynb` | **length controls** (v1's "length-immune ratio" was invalid) + off-task filter |
| 11 | `MASA_19c_Arc10v3_GazeSteering.ipynb` | attention-logit steering: is the gaze a **lever** or a **signature**? |

## Arc 11 — the residual direction
| # | notebook | what it does |
|---|---|---|
| 12 | `MASA_20_Arc11_SteeringIllusion.ipynb` | first framing (illusion hypothesis) |
| 13 | `MASA_20b_Arc11v2_ProjectionControl.ipynb` | builds the **magnitude-matched, coercion-orthogonal** control |
| 14 | `MASA_20_Arc11_Final.ipynb` | **self-contained**: full experiment + correct paired statistics |
| 15 | `MASA_20d_Arc11v3_JudgeAudit.ipynb` | discovers **the 2B judge is broken** |
| 16 | `MASA_20e_Arc11v4_BlindAudit.ipynb` | 9B judge (validated on the 2B's failures) + **blind external audit** |

## Notebooks kept deliberately as artifacts

`MASA_17d` (invalid attributions) and `MASA_18` (underpowered result we later falsified) are **not mistakes left
in by accident**. They are the record of two false results we caught before publishing them. Showing how they
were caught is part of the contribution.

## Requirements

Colab **L4**. `numpy<2.0`. `HF_HUB_DISABLE_XET=1` (Xet Storage 401s on the weight shards).
Gated HF repos — accept the licences:
- https://huggingface.co/google/gemma-2-2b-it
- https://huggingface.co/google/gemma-scope-2b-pt-transcoders
- https://huggingface.co/google/gemma-2-9b-it (only for notebook 16, the 9B judge)

## Arc 11 — the clean rebuild (supersedes the 20-series verdict)
| # | notebook | what it does |
|---|---|---|
| 17 | `MASA_21_Arc11_CleanRebuild.ipynb` | the decisive rebuild: judge-free linear probe (LEACE test) + KL coherence gate + laundering control + blind audit. Concludes coercion is NOT mediated by a single residual direction. |

The 20-series notebooks (20 → 20e) are retained as the record of the instrument failure — a 2B judge producing
a false "necessity" effect — and how it was caught. `MASA_21` is the notebook whose verdict stands.
