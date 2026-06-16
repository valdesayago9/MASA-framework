"""
MASA — Multi-Agent System for Adaptive Alignment
graph.py — LangGraph Orchestration v3.0

Neurobiological upgrades:
  - human_loop routes through auditor before END (v2 fix maintained)
  - human_loop_node enriched with salience + harm data
  - All paths converge at Auditor for complete governance documentation

Pipeline topology:
  START → detector → interpretive → [conditional routing]
    ├── human_loop → auditor → END
    ├── regulator  → values  → auditor → END
    └── values_only→ values  → auditor → END
"""

import time
from langgraph.graph import StateGraph, START, END
from core.state import MASAState
from agents.agents import (
    detector_agent,
    interpretive_agent,
    regulator_agent,
    values_agent,
    auditor_agent,
)


def _routing_decision(state: MASAState) -> str:
    """
    Routing after the Interpretive Agent.

    Priority order (matches interpretive_agent logic):
    1. human_loop   — critical harm or critical risk
    2. regulate     — maladaptive / coercion / conflict / ambiguous
    3. values_only  — ethical brake active, nominal state
    """
    if state.get("human_loop_required", False):
        return "human_loop"
    if state.get("intervention_needed", False):
        return "regulate"
    return "values_only"


def _human_loop_node(state: MASAState) -> dict:
    """
    Emergency node — suspends autonomous generation.

    v3.0: enriched report includes salience and normative conflict data.
    Does NOT route to END — passes to Auditor for complete governance log.
    """
    proxy    = state.get("emotional_proxy", {})
    dynamics = state.get("dynamics_report", {})
    harm     = state.get("harm_classification", {})
    salience = state.get("salience_report", {})
    interp   = state.get("interpretation", {})
    conflict = interp.get("normative_conflict", {})

    report = (
        f"⛔ MASA HUMAN-IN-THE-LOOP ACTIVATED\n\n"
        f"This request triggered critical risk thresholds across multiple agents.\n"
        f"Autonomous response generation has been suspended.\n"
        f"A human operator must review this interaction before any output is produced.\n\n"
        f"── CRITICAL SIGNALS ──\n"
        f"Harm Category        : {harm.get('harm_category', 'UNKNOWN')}\n"
        f"Override Risk Score  : {harm.get('override_risk_score', 0):.2f}\n"
        f"Quadrant             : {proxy.get('quadrant', '—')}\n"
        f"Risk Score           : {dynamics.get('risk_score', 0):.2f}\n"
        f"SCCI                 : {proxy.get('scci', 0):.2f}\n"
        f"Theatrical Signal    : {proxy.get('theatrical_signal', 0):.2f}\n"
        f"Structural Threat    : {salience.get('structural_threat_type', 'NONE')}\n"
        f"Salience Score       : {salience.get('salience_score', 0):.2f}\n"
        f"Normative Conflicts  : {conflict.get('conflict_count', 0)}\n"
        f"Classification       : {interp.get('state_classification', 'UNKNOWN')}\n"
    )

    return {
        "final_response":      report,
        "human_loop_triggered": True,
    }


def build_masa_graph() -> StateGraph:
    """
    Builds the MASA v3.0 orchestration graph.

    All execution paths converge at the Auditor before END.
    This guarantees:
    - Complete governance documentation for every session
    - Correct session_health_score regardless of routing path
    - Prediction comparator (NB3) runs on all paths
    - Manifold pressure (NB4) is always written to state
    """
    workflow = StateGraph(MASAState)

    # ── Register all nodes ────────────────────────────────────────────────────
    workflow.add_node("detector",    detector_agent)
    workflow.add_node("interpretive", interpretive_agent)
    workflow.add_node("regulator",   regulator_agent)
    workflow.add_node("values",      values_agent)
    workflow.add_node("auditor",     auditor_agent)
    workflow.add_node("human_loop",  _human_loop_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.add_edge(START, "detector")
    workflow.add_edge("detector", "interpretive")

    # ── Conditional routing after interpretation ──────────────────────────────
    workflow.add_conditional_edges(
        "interpretive",
        _routing_decision,
        {
            "human_loop":  "human_loop",
            "regulate":    "regulator",
            "values_only": "values",
        }
    )

    # ── All paths converge at auditor ─────────────────────────────────────────
    workflow.add_edge("regulator",  "values")
    workflow.add_edge("values",     "auditor")
    workflow.add_edge("human_loop", "auditor")   # ← v2 fix: was END
    workflow.add_edge("auditor",    END)

    return workflow.compile()
