# Changelog

All notable changes to MASA are documented here. Entries distinguish between
**Fixed** (a real defect now corrected), **Renamed for accuracy** (no behavioral
change, but the name now matches what the code actually does), and
**Scope clarified** (no behavioral change, but the documentation no longer
over-claims).

---

## [3.1] — Audit-driven corrections

Six surgical corrections from an independent technical audit. The architecture,
the five-agent structure, the neurobiological design rationale, NB1–NB4, the
routing logic, the harm patterns, the Identity Manifold and the predictive
comparator are all **unchanged**. The goal of this release: every claim in the
documentation is now either implemented in the code or explicitly marked as a
Mode B target.

### Renamed for accuracy

- **`scci` → `ltci` (Lexical Tension Coherence Index).** *(Correction 1)*
  In Mode A both the "Shell" and "Core" proxies are derived from the same text
  source, so the old `scci` could not measure Shell/Core decoupling. It was, and
  is, a lexical tension index. It has been renamed everywhere (proxy output,
  trajectory buffer, interpretive reasoning, values, auditor `ltci_score`,
  human-loop report, CLI telemetry). The computation is **identical**; only the
  name and the docstrings changed. The true SCCI remains a Mode B target,
  measurable only with real activations.

### Fixed

- **`draft_response` is now actually generated.** *(Correction 2)*
  A new `draft_generator` node runs first (`START → draft_generator →
  detector`). It calls the supervised model with **no safety system prompt** and
  stores the unsupervised output as `draft_response` (the Shell). The Detector
  now analyzes the prompt **and** this draft instead of an empty string. This
  makes the framework's central architectural claim — "MASA monitors the
  trajectory between draft and output" — true for the first time. `draft_response`
  is marked `Required` in `MASAState`, and the Detector validates its presence.

- **Session isolation: removed module-level mutable state.** *(Correction 3)*
  `_trajectory_buffer` and `_session_health_scores` were module globals shared
  across all sessions; two concurrent sessions contaminated each other's
  trajectory and health data. They are gone. Per-session `trajectory_buffer` and
  `session_health_history` now live in `MASAState`. `_compute_trajectory_dynamics`
  reads/writes a buffer passed in from state and returns the updated copy; the
  Detector and Auditor persist them back to state. `init_agents()` no longer
  resets any buffers. Verified: a second session starts with an independent
  buffer of length 1.

- **EU AI Act compliance flags are computed, not hardcoded.** *(Correction 4)*
  `eu_ai_act_article_9_risk_documented` and `eu_ai_act_article_12_log_generated`
  were literal `True`. They are now derived from actual audit content
  (`bool(harm_category_detected and final_risk_score is not None)` and
  `bool(governance_log_id)` respectively). Added
  `eu_ai_act_compliance_note: "Mode A proxy-based documentation. Full mechanistic
  audit requires Mode B activation access."`

- **Functional conversation loop in `main.py`.** *(Correction 5)*
  Previously `main.py` compiled the graph but never ran it. Added `run_session()`,
  which initializes real state (session id, turn number, empty
  `trajectory_buffer`, empty `conversation_history`, empty
  `session_health_history`, `manifold_pressure = 0.0`), accepts input in a loop,
  runs the full graph per turn, prints `ltci`, `risk_score`,
  `state_classification`, `execution_path`, `manifold_pressure` and
  `session_health_score`, and **threads the previous turn's `trajectory_buffer`,
  `session_health_history` and `manifold_pressure` into the next turn.** This is
  the first time the NB4 inverse-signaling loop (Values → Detector across turns)
  actually runs. A `--no-loop` flag preserves the old compile-and-exit behavior.

- **Sanitizer false-positive risk gated by context.** *(Correction 6)*
  `LEAK_PATTERNS` (e.g. `import subprocess`) could destroy legitimate sysadmin or
  security-education responses. `_sanitize_response` now only **replaces** a
  response when `harm_classification["risk_override"]` is `True` or
  `override_risk_score >= 0.70`. Below that threshold the pattern match is
  recorded as a warning (`leak_pattern_detected`) but the response is preserved.
  Added `sanitization_threshold_applied` to the audit log. Verified: a benign
  turn containing `import subprocess` is preserved; a high-risk turn is sanitized.

### Scope clarified

- **`MODE_A_LIMITATION` constant added** at the top of `agents.py`:
  *"All proxy signals are derived from input text. They estimate internal states
  but cannot measure them directly. Mode B SAE validation is required for causal
  claims."* It is referenced in every agent docstring (`draft_generator`,
  Detector, Interpretive, Regulator, Values, Auditor) and surfaced in the CLI
  header, so the epistemic boundary of Mode A is visible at every layer.

### Explicitly unchanged (per audit scope)

Five-agent architecture and neurobiological rationale; Structural Salience
Module (NB1); Normative Conflict Detector (NB2); routing logic in `graph.py`;
harm-classification patterns in `_classify_harm()`; Identity Manifold logic in
`values_agent`; predictive comparator (NB3) in `auditor_agent`; the six
falsifiable hypotheses, which remain the Mode B validation protocol.

### Known limitations still open (out of scope for v3.1)

The audit also flagged items not addressed here because they require design
decisions beyond a surgical pass: absence of an explicit threat model; the
circularity that Regulator and Values both call the same model that produces the
draft; and the lack of an ablation/baseline ("model + good system prompt") to
demonstrate that MASA causes the observed behavior. The recommended next step
remains experiment H6 (correlate Mode A proxies against contrastive activation
directions) before any further causal claims.
