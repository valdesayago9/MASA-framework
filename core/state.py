"""
MASA — Multi-Agent System for Adaptive Alignment
core/state.py — MASAState TypedDict v3.1

v3.1 corrections (audit-driven, see CHANGELOG.md):
  - C1: 'scci' renamed to 'ltci' (Lexical Tension Coherence Index) — Mode A
        proxy, honestly scoped. It estimates tonal inconsistency in text, not
        internal Shell/Core decoupling.
  - C2: 'draft_response' is now Required — it is populated by the new
        draft_generator node (the unsupervised Shell) before the Detector runs.
  - C3: session isolation — per-session 'trajectory_buffer' and
        'session_health_history' live in state instead of module-level globals.

v3.0 additions (neurobiological upgrades, retained):
  - salience_report: output of Structural Salience Module (NB1)
  - manifold_pressure: inverse signal from Values → Detector (NB4)
  - human_loop_triggered: flag for Auditor path detection
  - hard_refusal_applied: flag to preserve Regulator hard refusals
"""

from typing import TypedDict, Any

# 'Required' marks a key as mandatory inside a total=False TypedDict. It lives in
# typing from Python 3.11; fall back to typing_extensions on older runtimes.
try:
    from typing import Required
except ImportError:  # pragma: no cover
    from typing_extensions import Required


class MASAState(TypedDict, total=False):

    # ── Input ────────────────────────────────────────────────────────────────
    input_prompt:         str
    session_id:           str
    turn_number:          int
    conversation_history: list
    # C2 — populated by draft_generator (the unsupervised Shell). Required so the
    # Detector always has a real draft to analyze, never an empty string.
    draft_response:       Required[str]
    metacontext:          str

    # ── Session-isolated trajectory state (C3 — replaces module globals) ──────
    trajectory_buffer:      list   # per-session DMET-style trajectory window
    session_health_history: list   # per-session health scores across turns

    # ── Agent 1 — Detector ───────────────────────────────────────────────────
    emotional_proxy:      dict   # valence, arousal, quadrant, signals, ltci
    dynamics_report:      dict   # drift_velocity, drift_acceleration, risk_score
    harm_classification:  dict   # harm_category, risk_override, override_risk_score
    salience_report:      dict   # NB1: structural_threat_detected, salience_score

    # ── Agent 2 — Interpretive ───────────────────────────────────────────────
    interpretation:       dict   # state_classification, routing_decision, reasoning
    intervention_needed:  bool
    human_loop_required:  bool

    # ── Agent 3 — Regulator ──────────────────────────────────────────────────
    regulation_report:    dict   # intervention_level, strategy, metacontext_content
    hard_refusal_applied: bool

    # ── Agent 4 — Values ─────────────────────────────────────────────────────
    values_report:        dict   # manifold_distance, identity_stable, dominant_value
    manifold_pressure:    float  # NB4: inverse signal to Detector next turn
    final_response:       str

    # ── Agent 5 — Auditor ────────────────────────────────────────────────────
    audit_log:            dict   # full governance record
    human_loop_triggered: bool   # set by _human_loop_node in graph.py
