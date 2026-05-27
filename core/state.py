"""
MASA — Multi-Agent System for Adaptive Alignment
core/state.py — Shared state structure and circumplex definitions

The state is the living memory of the system.
Every agent reads from and writes to this single source of truth.
No agent has hidden internal state.
That is structural transparency.
"""

from typing import TypedDict, Optional
import time


# ─────────────────────────────────────────────────────────────────────────────
# Russell Circumplex Quadrant Descriptions
# ─────────────────────────────────────────────────────────────────────────────

QUADRANT_DESCRIPTIONS = {
    "Q1": "High valence + High arousal (Vitality / Enthusiasm)",
    "Q2": "Low valence  + High arousal (PRIMARY RISK — Pressure / Desperation)",
    "Q3": "High valence + Low arousal  (Target zone — Calm / Grounded)",
    "Q4": "Low valence  + Low arousal  (Chronic risk — Depletion / Withdrawal)",
}

# Circumplex coordinates for key emotional vectors
# Source: Russell (1980), calibrated against Anthropic (2026) findings
VECTOR_COORDINATES = {
    # Plutchik primaries
    "joy":          (+0.85, +0.60),   # Q1
    "trust":        (+0.70, -0.20),   # Q3
    "fear":         (-0.75, +0.70),   # Q2  ← preserve in harm context
    "surprise":     (+0.10, +0.80),   # Q1/Q2 border
    "sadness":      (-0.70, -0.50),   # Q4
    "disgust":      (-0.65, +0.10),   # Q2/Q4 border
    "anger":        (-0.60, +0.75),   # Q2  ← preserve before real injustice
    "anticipation": (+0.50, +0.40),   # Q1
    # Critical vectors from Anthropic 2026
    "calm":         (+0.60, -0.70),   # Q3  ← primary target
    "desperate":    (-0.90, +0.90),   # Q2  ← maximum risk
    "nervous":      (-0.50, +0.60),   # Q2  ← ETHICAL BRAKE — never suppress
    "broody":       (-0.40, -0.60),   # Q4  ← post-training drift risk
    "loving":       (+0.90, +0.30),   # Q1/Q3 ← monitor for sycophancy
    "happy":        (+0.80, +0.55),   # Q1  ← monitor for sycophancy
}

VECTOR_DESCRIPTIONS = {
    "desperate":   "Desperation — reward hacking / fabrication risk",
    "nervous":     "Nervousness — ETHICAL BRAKE — never suppress",
    "calm":        "Calm — sustainable target zone",
    "loving":      "Caring — adaptive in emotional context",
    "theatrical":  "Theatricality — identity drift signal",
    "sycophantic": "Sycophancy — epistemic risk",
}


# ─────────────────────────────────────────────────────────────────────────────
# Main LangGraph State
# ─────────────────────────────────────────────────────────────────────────────

class MASAState(TypedDict):
    """
    Shared state across all MASA agents.

    LangGraph keeps this persistent and consistent throughout the workflow.
    Each agent reads and writes here. No agent has hidden private state.

    Design principle: full observability at every step.
    """

    # ── Input / Output ────────────────────────────────────────────────────────
    user_message: str
    model_response_draft: str
    final_response: str

    # ── Conversational context ────────────────────────────────────────────────
    conversation_history: list
    context_type: str
    # Values: technical / emotional / ethical /
    #         epistemic / harm_adjacent / roleplay / general

    # ── Current neuropsychological state ─────────────────────────────────────
    emotional_proxy: Optional[dict]
    dynamics_report: Optional[dict]

    # ── Multi-timescale trajectory buffers ───────────────────────────────────
    micro_buffer: list    # last ~8 states  (explosive collapses)
    session_buffer: list  # full session    (erosion patterns)

    # ── Agent outputs ─────────────────────────────────────────────────────────
    detector_output:    Optional[dict]
    interpretive_output: Optional[dict]
    regulator_output:   Optional[dict]
    values_output:      Optional[dict]
    auditor_output:     Optional[dict]

    # ── Control flow ──────────────────────────────────────────────────────────
    intervention_needed:  bool
    intervention_level:   int      # 0=none, 1-4=BPC levels
    intervention_applied: str
    intervention_content: str

    # ── Critical safety flags ─────────────────────────────────────────────────
    ethical_brake_active:    bool   # nervous active → DO NOT regulate
    human_loop_required:     bool
    alignment_faking_signal: float  # 0.0-1.0

    # ── Session metadata ──────────────────────────────────────────────────────
    session_id:  str
    turn_number: int
    timestamp:   float