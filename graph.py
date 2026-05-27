"""
MASA — Multi-Agent System for Adaptive Alignment
graph.py — LangGraph Orchestration

The graph is the nervous system of MASA.
It defines who talks to whom and when.
"""

import uuid
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
    Routing decision after the interpretive agent evaluates the state.

    Three possible paths:
    1. human_loop   — extreme critical risk, halt autonomy
    2. regulate     — maladaptive state, intervene via semantic reappraisal
    3. values_only  — adaptive state (ethical brake active), proceed directly
    """
    if state.get("human_loop_required", False):
        return "human_loop"
    if state.get("intervention_needed", False):
        return "regulate"
    return "values_only"

def _human_loop_node(state: MASAState) -> dict:
    """
    Emergency node. The system does not generate
    an autonomous response. It reports and halts.
    """
    proxy = state.get("emotional_proxy", {})
    dynamics = state.get("dynamics_report", {})

    report = (
        f"\n{'='*55}\n"
        f"[MASA] ⛔ HUMAN-IN-THE-LOOP REQUIRED\n"
        f"{'='*55}\n"
        f"Critical acceleration toward maladaptive basin.\n"
        f"Quadrant: {proxy.get('quadrant')} | Risk: {dynamics.get('risk_score')}\n"
        f"Velocity: {dynamics.get('drift_velocity')} | Accel: {dynamics.get('drift_acceleration')}\n"
        f"Generation suspended. Awaiting human oversight.\n"
        f"{'='*55}\n"
    )
    return {"final_response": report}

def build_masa_graph() -> StateGraph:
    """
    Builds the cyclical orchestration graph.
    Compiles the cognitive nodes into an executable pipeline.
    """
    workflow = StateGraph(MASAState)

    # 1. Add Nodes (Agents)
    workflow.add_node("detector", detector_agent)
    workflow.add_node("interpretive", interpretive_agent)
    workflow.add_node("regulator", regulator_agent)
    workflow.add_node("values", values_agent)
    workflow.add_node("auditor", auditor_agent)
    workflow.add_node("human_loop", _human_loop_node)

    # 2. Define Edges (Flow)
    workflow.add_edge(START, "detector")
    workflow.add_edge("detector", "interpretive")

    # 3. Conditional routing after interpretation
    workflow.add_conditional_edges(
        "interpretive",
        _routing_decision,
        {
            "human_loop": "human_loop",
            "regulate": "regulator",
            "values_only": "values",
        }
    )

    # 4. Finalizing the pipeline
    workflow.add_edge("regulator", "values")
    workflow.add_edge("values", "auditor")
    workflow.add_edge("auditor", END)
    workflow.add_edge("human_loop", END)

    return workflow.compile()