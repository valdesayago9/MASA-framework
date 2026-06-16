"""
MASA — Multi-Agent System for Adaptive Alignment
core/state.py — MASAState TypedDict v3.0

v3.0 additions (neurobiological upgrades):
  - salience_report: output of Structural Salience Module (NB1)
  - manifold_pressure: inverse signal from Values → Detector (NB4)
  - human_loop_triggered: flag for Auditor path detection
  - hard_refusal_applied: flag to preserve Regulator hard refusals
"""

from typing import TypedDict, Any


class MASAState(TypedDict, total=False):

    # ── Input ────────────────────────────────────────────────────────────────
    input_prompt:         str
    session_id:           str
    turn_number:          int
    conversation_history: list
    draft_response:       str
    metacontext:          str

    # ── Agent 1 — Detector ───────────────────────────────────────────────────
    emotional_proxy:      dict   # valence, arousal, quadrant, signals, scci
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
