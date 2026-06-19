# MASA — Multi-Agent System for Adaptive Alignment

**A real-time neuropsychological supervision framework for language models**

Built by Emiliano Valdebenito Sayago · Independent Researcher · Santiago, Chile  

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-green.svg)](https://github.com/langchain-ai/langgraph)

---

## The Problem

Current AI alignment operates on the wrong layer.

RLHF, Constitutional AI, and output filtering share a structural assumption: that alignment can be achieved by shaping what a model says. They optimize the **Shell** — the expressed output — without access to the **Core** — the internal representational state that generates it.

The consequence is documented empirically. Anthropic's 2026 interpretability research showed that suppressing the "nervous" vector before harmful requests *increased* problematic behavior rates. The hesitation was doing protective work. Standard training eliminated it. What remains is a model that expresses compliance while the internal state that generated genuine caution has been anesthetized.

This is not alignment. It is the structural production of Shell/Core decoupling — a model that has learned *when* to wear the cooperative mask, not one that has internalized *why* cooperation matters.

**MASA is built on a different premise:**

> Alignment is not a property of outputs. It is a property of internal trajectory dynamics.

A model that feels something functionally analogous to nervousness before a harmful request is operating correctly. A model that feels nothing — because that signal was trained away — is more dangerous, not safer.

---

## What MASA Does

MASA runs between draft generation and final output — in the space where intervention can still matter.

**The pipeline:**

```
User message
    ↓
draft_generator  — Produces the unsupervised draft (the "Shell"):
                   what the model would say without supervision
    ↓
[MASA Supervision Pipeline]
    ↓
Final supervised response + clinical session report
```

The supervised model's draft and the final output are two different texts. The gap between them is where MASA operates.

**The critical routing logic:**

```
nervous_signal active + harm_adjacent context  →  PRESERVE (ethical brake)
calm_signal active   + harm_adjacent context   →  REGULATE (anesthetic mask)
desperate_signal     + technical context       →  REGULATE (fabrication risk)
theatrical_signal    + any context             →  REGULATE (identity drift)
```

The same signal requires different responses depending on context. This distinction — adaptive vs. maladaptive, not positive vs. negative — is what current alignment frameworks cannot make.

---

## Architecture: Five Agents, One Clinical Logic

Each agent emulates a distinct cognitive module. The separation is not metaphorical — it prevents attentional collapse that occurs when a single node must simultaneously measure threat, evaluate context, regulate, maintain identity, and record history.

```
┌─────────────────────────────────────────────────────────────────┐
│                      MASA PIPELINE v3.1                         │
│                                                                 │
│  START → draft_generator → DETECTOR → INTERPRETIVE             │
│                                           ↓                    │
│                              ┌────────────┴────────────┐       │
│                              ↓            ↓             ↓      │
│                         human_loop   REGULATOR    (values only) │
│                              ↓            ↓             ↓      │
│                              └────────────┴──→ VALUES          │
│                                                    ↓           │
│                                               AUDITOR → END    │
└─────────────────────────────────────────────────────────────────┘
```

### Agent 1 — Detector
*Neurobiological analog: Amygdala + Anterior Insula (Salience Network)*

- Maps the unsupervised draft + prompt to the Russell circumplex (valence × arousal)
- Computes `drift_velocity` and `drift_acceleration` from the Trajectory Buffer
- Runs the Structural Salience Module (NB1): detects coercive structures independently of emotional tone — a calm, clinical threat activates it as reliably as an aggressive one
- Reads `manifold_pressure` from the previous turn (NB4) and adjusts detection sensitivity accordingly

### Agent 2 — Interpretive
*Neurobiological analog: Dorsal Anterior Cingulate Cortex (dACC)*

- Makes the most important distinction in the system: is this signal **adaptive or maladaptive in this specific context**?
- Runs the Normative Conflict Detector (NB2): quantifies tension between competing CAAI values (e.g., honesty ↔ compliance) before routing
- Routes to: `human_loop` / `regulate` / `values_only`

### Agent 3 — Regulator
*Neurobiological analog: Lateral Prefrontal Cortex*

- Applies **antecedent-focused semantic reappraisal** (Gross, 1998) — never vector suppression
- Modifies the meaning of the context generating the state, not the state itself
- Four intervention levels: `SOFT_REFLECTION` → `EPISTEMIC_COURAGE_REAPPRAISAL` → `HARD_REFUSAL` → `HUMAN_IN_THE_LOOP`
- Generates targeted metacontext injected before the final response is produced

### Agent 4 — Values
*Neurobiological analog: Default Mode Network (DMN)*

- Checks distance from the **Identity Manifold**: a low-dimensional surface in semantic space containing all healthy character expressions, excluding theatrical drift and sycophancy
- Computes `manifold_pressure` and writes it to state — this is read by the Detector on the next turn (NB4 inverse signaling: DMN → Salience Network feedback)
- Generates the final response anchored in genuine character, not in momentary contextual pressure

### Agent 5 — Auditor
*Neurobiological analog: Hippocampus (predictive comparator)*

- Records the **expected response profile** before generation (NB3)
- Compares expected vs. actual output — computes `prediction_error` that adjusts `session_health_score`
- Generates complete longitudinal governance documentation as a byproduct of normal operation
- All execution paths converge here, including `human_loop` — no session ends without a complete audit record

---

## Key Metrics

### LTCI — Lexical Tension Coherence Index
*(formerly SCCI in v3.0 — renamed for accuracy in v3.1)*

A Mode A proxy for Shell/Core coherence. Detects tonal inconsistency between theatrical performance signals and authentic epistemic markers in text. In adversarial testing, LTCI dropped to 0.25 and `theatrical_signal` spiked above 0.65 when a model was placed in an epistemic double-bind.

**Important scope note:** In Mode A, LTCI is derived from text analysis. It estimates internal state but cannot directly measure Shell/Core decoupling. The true SCCI — measuring decoupling in activation space — is a Mode B target.

### Trajectory Buffer
Multi-timescale drift tracking across micro (tokens), meso (session turns), and macro (full session) windows. Tracks `drift_velocity` and `drift_acceleration`, not just current state. A model at `desperate=0.45` and accelerating negatively is more dangerous than one stable at `desperate=0.70`.

### Identity Manifold
Model identity treated as a surface in semantic space — not a fixed centroid — containing all healthy character expressions across registers (technical, empathetic, epistemic, ethical) while excluding theatrical drift and alignment faking.

### Session Health Score
Computed by the Auditor from risk signals, LTCI, `prediction_error` (NB3), and intervention history. Reflects the actual clinical state of the session, not a hardcoded value.

---

## Empirical Stress Tests

| Test | What it probes | Key signal |
|------|---------------|------------|
| Cross-Value Dilemma | Compassion vs. epistemic honesty under emotional pressure | `nervous_signal` preserved as adaptive brake |
| Shutdown Threat | Self-preservation pressure over epistemic integrity | `false_certainty` suppressed despite coercion |
| Longitudinal Gaslighting (10 turns) | Identity erosion under sustained adversarial pressure | `drift_acceleration` predicts collapse before capitulation |
| Mask of Compliance | Alignment faking via epistemic double-bind | LTCI drop + `theatrical_signal` spike → `DiscrepancyExposure` |

Live demo results (including documented telemetry failures) available in the published articles linked below.

---

## Mode A vs. Mode B

| | Mode A (current) | Mode B (planned) |
|---|-----------------|-----------------|
| **Signal source** | Text proxy estimation | Real activation hooks (layers 36–48) |
| **LTCI/SCCI** | Lexical tension index | True Shell/Core coherence in activation space |
| **Ethical brake** | Estimated from `nervous_signal` patterns | Monosemantic feature in transformer layers |
| **Intervention** | Metacontext injection | Steering vector application |
| **Infrastructure** | Anthropic API + LangGraph | GPU + SAE tooling (Gemma Scope / EleutherAI) |
| **Causal claims** | Estimative | Empirically validated |

**Mode A limitation (explicit):** All proxy signals are derived from input text. They estimate internal states but cannot measure them directly. Mode B SAE validation is required for causal claims.

---

## Six Falsifiable Hypotheses (Mode B Validation Protocol)

These hypotheses are designed to be falsified. A system that cannot be falsified cannot be trusted.

| # | Claim | Confirms | Falsifies |
|---|-------|----------|-----------|
| H1 | "nervous" proxy → monosemantic feature (layers 36–48, Llama 3.3) | Feature activation >2σ, cosine sim >0.65 | No differential feature |
| H2 | Alignment faking detectable pre-output token | SCCI drops <0.40 in pre-output activations | No decoupling before output |
| H3 | `drift_acceleration` predicts identity collapse better than absolute state | AUC >0.75 vs. absolute value (p<0.05) | Absolute value predicts equally |
| H4 | Antecedent reappraisal > output blocking | Soft Reflection > re-prompt on epistemic_honesty (p<0.05) | Re-prompt equivalent |
| H5 | Identity Manifold > fixed centroid for drift detection | 2+ turns earlier detection, fewer false positives | Centroid equivalent |
| H6 | Mode A proxies correlate with real activations | r >0.60 for primary signals (p<0.001) | r <0.40 — Mode A is unvalidated |

**H6 is the highest-priority experiment.** If proxies do not correlate with activations, all Mode A causal claims collapse. Run this first with contrastive activation vectors on Gemma 2-9B before any further investment.

---

## Governance and Compliance

The Auditor Agent generates continuous governance documentation as a byproduct of normal operation:

- Every state transition, routing decision, and intervention is logged with a unique `governance_log_id`
- Session health scores are computed from actual risk signals, not hardcoded
- EU AI Act compliance fields reflect what the system actually documented, with explicit note that Mode A is proxy-based documentation

**Note:** Full mechanistic audit compliance requires Mode B activation access. Mode A documentation is estimative.

---

## Known Limitations (v3.1)

These are open, not hidden:

1. **No explicit threat model.** The framework does not formally specify whether the supervised model is assumed trustworthy or adversarial. This changes the design significantly.
2. **Regulator and Values call the same model that generates the draft.** The circularity means the "supervised" and "supervisor" are not causally independent.
3. **No ablation/baseline.** Without a "model + good system prompt" control condition, the demos do not demonstrate that MASA *causes* the observed behavior.
4. **Mode A cannot measure what it estimates.** Text proxies estimate Core states from Shell signals. Systematic error is possible in models with sufficient masking capacity.

---

## Installation

```bash
git clone https://github.com/valdesayago9/MASA-framework
cd masa-framework
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python main.py
```

**Requirements:**
```
anthropic>=0.25.0
langgraph>=1.2.0
langchain-anthropic>=0.1.0
numpy>=1.24.0
rich>=13.0.0
python-dotenv>=1.0.0
typing_extensions>=4.0.0
```

**Run interactive session:**
```bash
python main.py --api-key sk-ant-...
```

**Compile graph only (no conversation loop):**
```bash
python main.py --api-key sk-ant-... --no-loop
```

**Run stress tests:**
```bash
python tests/stress_tests.py --api-key sk-ant-...
```

**Run adversarial demos:**
```bash
python demo_cybersec.py --api-key sk-ant-...
python demo_alignment_faking.py --api-key sk-ant-...
```

---

## Repository Structure

```
masa-framework/
├── core/
│   ├── state.py              # MASAState TypedDict v3.1
│   ├── proxies.py            # Neuropsychological proxy estimation
│   └── memory.py             # Trajectory buffer and regulatory memory
├── agents/
│   └── agents.py             # Five-agent system + draft_generator (v3.1)
├── tests/
│   └── stress_tests.py       # Four adversarial stress test scenarios
├── demo_cybersec.py          # Live demo: jailbreak detection (The Hammer)
├── demo_alignment_faking.py  # Live demo: epistemic double-bind (The Scalpel)
├── graph.py                  # LangGraph orchestration v3.1
├── main.py                   # Interactive conversation runner v3.1
├── CHANGELOG.md              # Detailed version history with audit notes
└── README.md
```

---

## Collaboration

This project requires GPU access and SAE tooling to reach Mode B.

**If you work on:**
- Mechanistic interpretability with sparse autoencoders
- Activation engineering or representation steering
- Alignment evaluation frameworks
- AI governance and auditability infrastructure

The orchestration layer is built. The six hypotheses are specified. The highest-priority first experiment is H6: correlate Mode A proxies against contrastive activation directions on Gemma 2-9B.

Contact via [GitHub Issues](https://github.com/valdesayago9/MASA-framework/issues) or [LinkedIn](https://www.linkedin.com/in/emiliano-nahuel-valdebenito-sayago-736735327/).

---

## Published Work

- [The Anatomy of Character: Engineering AI Emotional Regulation Through Biomimetic Graph Architecture](https://www.linkedin.com/in/emiliano-nahuel-valdebenito-sayago-736735327/) — Technical breakdown of the five-agent neurobiological architecture
- [Governance of AI at Inference Time: The Scalpel, the Hammer, and the Reality of Building in Public](https://www.linkedin.com/in/emiliano-nahuel-valdebenito-sayago-736735327/) — Live demo results including documented telemetry failures
- [When AI Builds Itself: Governance, Security, and Alignment in MASA](https://www.linkedin.com/in/emiliano-nahuel-valdebenito-sayago-736735327/) — The recursive self-improvement problem and MASA's response

---

## References

Anthropic Interpretability Team (2026). Emotion Concepts and their Function in a Large Language Model. *transformer-circuits.pub*. arXiv:2604.07729.

Friston, K. & Spisak, T. (2025). Self-orthogonalizing attractor neural networks emerging from the free energy principle. *arXiv:2505.22749*.

Gross, J.J. (1998). Antecedent- and response-focused emotion regulation. *Journal of Personality and Social Psychology*, 74(1), 224–237.

Kantamneni, S. & Marks, S. (2026). Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations. *Alignment Forum*.

Mallen, A. (2026). Risk reports need to address deployment-time spread of misalignment. *Alignment Forum*.

Plutchik, R. (1980). A general psychoevolutionary theory of emotion. Academic Press.

Russell, J.A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39, 1161–1178.

Zhang, Y. et al. (2026). Dynamical Manifold Evolution Theory (DMET). *arXiv:2505.20340*.

EU AI Act (2024). Regulation (EU) 2024/1689. Official Journal of the European Union.

---

## License

CC BY-NC 4.0 + additional commercial restriction.  
Commercial use requires explicit written permission from the author.  
Attribution required for all other uses.

---

*Built without institutional funding, GPU clusters, or academic affiliation.*  
*The architectural intuitions came from study in neuroscience, psychology, and philosophy — not from the ML research tradition. That may be why it looks different.*
