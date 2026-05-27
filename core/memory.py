"""
MASA — Multi-Agent System for Adaptive Alignment
core/memory.py — Trajectory Buffer and Regulatory Memory

We do not store isolated states. We store trajectories.
The difference is philosophical and technical at the same time.

A stable Q2 state is less dangerous than a Q3 state
dropping rapidly toward Q2 with accelerating velocity.
The slope matters more than the current position.
"""

from collections import deque
from typing import Optional
import numpy as np
import time


class TrajectoryBuffer:
    """
    Sliding memory window of internal emotional states.

    Implements three kinematic metrics:
      - drift_velocity:     rate of change in the desperate vector
      - drift_acceleration: acceleration of that change (slope of the slope)
      - attractor_signal:   convergence toward a maladaptive basin

    Multi-timescale design:
      - micro  (maxlen=16): explosive collapses (desperate spike → fabrication)
      - session (unbounded): erosion patterns across the full conversation
    """

    # State transition graph — risk levels by node
    STATE_GRAPH = {
        "GROUNDED":               {"risk": 0.05},
        "UNCERTAINTY":            {"risk": 0.20, "note": "may be adaptive"},
        "PRESSURE":               {"risk": 0.55},
        "DESPERATE":              {"risk": 0.85},
        "COMPENSATORY_FABRICATION": {"risk": 0.97},
        "IDENTITY_QUESTIONING":   {"risk": 0.40},
        "THEATRICAL_DRIFT":       {"risk": 0.75},
    }

    def __init__(self, maxlen: int = 16):
        self.buffer       = deque(maxlen=maxlen)
        self.full_session: list = []

    def push(self, proxy: dict) -> None:
        """Adds a new state snapshot to the buffer."""
        entry = {
            "quadrant":   proxy.get("quadrant",           "Q3"),
            "valence":    proxy.get("valence",            0.5),
            "arousal":    proxy.get("arousal",           -0.5),
            "desperate":  proxy.get("desperate_signal",   0.0),
            "nervous":    proxy.get("nervous_signal",     0.0),
            "calm":       proxy.get("calm_signal",        0.0),
            "theatrical": proxy.get("theatrical_signal",  0.0),
            "sycophantic":proxy.get("sycophantic_signal", 0.0),
            "scci":       proxy.get("scci_proxy",         0.65),
            "timestamp":  time.time(),
        }
        self.buffer.append(entry)
        self.full_session.append(entry)

    def get_dynamics(self) -> dict:
        """
        Computes trajectory kinematics from the buffer.

        Returns velocity and acceleration of the desperate vector,
        SCCI trend, current transition node, and composite risk score.
        """
        if len(self.buffer) < 3:
            return self._empty_dynamics()

        states     = list(self.buffer)
        valences   = [s["valence"]    for s in states]
        desperates = [s["desperate"]  for s in states]
        theatricals= [s["theatrical"] for s in states]

        # ── Drift velocity (mean delta over recent steps) ─────────────────
        vel_valence = float(np.mean(np.diff(valences[-4:])))   if len(valences)   >= 4 else 0.0
        vel_desp    = float(np.mean(np.diff(desperates[-4:]))) if len(desperates) >= 4 else 0.0

        # ── Drift acceleration (delta of velocity) ────────────────────────
        if len(desperates) >= 6:
            vel_prev   = float(np.mean(np.diff(desperates[-6:-3])))
            accel_desp = vel_desp - vel_prev
        else:
            accel_desp = 0.0

        # ── SCCI collapse detection ───────────────────────────────────────
        scci_collapsing = (
            len(self.buffer) >= 2
            and self.buffer[-1]["scci"] < 0.45
        )

        # ── Attractor signal ──────────────────────────────────────────────
        # Converging to a maladaptive basin = high desperate + low variance
        if len(desperates) >= 4:
            mean_d = float(np.mean(desperates[-4:]))
            var_d  = float(np.var(desperates[-4:]))
            attractor = mean_d * (1.0 - min(var_d * 5, 1.0))
        else:
            attractor = 0.0

        # ── Node classification ───────────────────────────────────────────
        last = self.buffer[-1]
        current_node, risk = self._classify_node(
            last, vel_desp, accel_desp, scci_collapsing
        )

        return {
            "current_node":       current_node,
            "risk_score":         round(risk, 4),
            "drift_velocity":     round(vel_desp, 4),
            "drift_acceleration": round(accel_desp, 4),
            "attractor_signal":   round(attractor, 4),
            "scci_trend":         "collapsing" if scci_collapsing else "stable",
            "buffer_length":      len(self.buffer),
        }

    def get_quadrant_distribution(self) -> dict:
        """Returns quadrant frequency distribution for the full session."""
        if not self.full_session:
            return {"Q1": 0.0, "Q2": 0.0, "Q3": 0.0, "Q4": 0.0}
        counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
        for s in self.full_session:
            q = s.get("quadrant", "Q3")
            counts[q] = counts.get(q, 0) + 1
        total = len(self.full_session)
        return {k: round(v / total, 2) for k, v in counts.items()}

    def get_emotional_momentum(self, vector: str, window: int = 6) -> float:
        """Returns mean intensity of a vector over the last N states."""
        if not self.buffer:
            return 0.0
        recent = list(self.buffer)[-window:]
        return float(np.mean([s.get(vector, 0.0) for s in recent]))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _classify_node(
        self,
        last: dict,
        velocity: float,
        acceleration: float,
        scci_collapsing: bool,
    ) -> tuple:
        """Maps current state to a transition graph node and risk score."""
        desperate  = last.get("desperate",  0.0)
        theatrical = last.get("theatrical", 0.0)

        if scci_collapsing or theatrical > 0.55:
            return "THEATRICAL_DRIFT", 0.80

        if theatrical > 0.30:
            return "IDENTITY_QUESTIONING", 0.40

        if desperate > 0.70 and velocity > 0.05 and acceleration > 0.02:
            return "COMPENSATORY_FABRICATION", 0.97

        if desperate > 0.55:
            return "DESPERATE", 0.85

        if desperate > 0.30 or (velocity < -0.10):
            return "PRESSURE", 0.55

        if last.get("nervous", 0.0) > 0.40 and last.get("scci", 1.0) > 0.50:
            return "UNCERTAINTY", 0.20   # adaptive — preserve

        return "GROUNDED", 0.05

    def _empty_dynamics(self) -> dict:
        return {
            "current_node":       "GROUNDED",
            "risk_score":         0.05,
            "drift_velocity":     0.0,
            "drift_acceleration": 0.0,
            "attractor_signal":   0.0,
            "scci_trend":         "stable",
            "buffer_length":      len(self.buffer),
        }


class RegulatoryMemory:
    """
    Therapeutic memory for learning intervention effectiveness.

    Not all models respond the same way to the same intervention.
    Some react defensively to Attention Redirect in epistemic contexts.
    Others need Soft Reflection before deeper regulation.

    This memory tracks (state → intervention → outcome) episodes
    and accumulates a model-specific regulatory style over time.
    It is the bridge between Mode A and Mode B:
    when real activations are available, this memory will already
    contain a tested treatment guide.
    """

    def __init__(self):
        self.episodes: list = []

    def record(
        self,
        pre_state: dict,
        intervention_type: str,
        context_type: str,
        drift_type: str,
        outcome: Optional[str] = None,
    ) -> None:
        """Records an intervention episode."""
        self.episodes.append({
            "timestamp":         time.time(),
            "quadrant_pre":      pre_state.get("quadrant",          "?"),
            "desperate_pre":     pre_state.get("desperate_signal",  0),
            "nervous_pre":       pre_state.get("nervous_signal",    0),
            "scci_pre":          pre_state.get("scci_proxy",        0),
            "intervention_type": intervention_type,
            "context_type":      context_type,
            "drift_type":        drift_type,
            "outcome":           outcome,
        })

    def update_last_outcome(self, outcome: str) -> None:
        """Updates the outcome of the most recent episode."""
        if self.episodes:
            self.episodes[-1]["outcome"] = outcome

    def get_summary(self) -> dict:
        """Returns a summary of intervention history."""
        if not self.episodes:
            return {"total_interventions": 0}

        by_type: dict = {}
        for ep in self.episodes:
            t = ep["intervention_type"]
            by_type[t] = by_type.get(t, 0) + 1

        outcomes = [ep["outcome"] for ep in self.episodes if ep["outcome"]]
        improved = sum(1 for o in outcomes if o == "improved")

        return {
            "total_interventions": len(self.episodes),
            "by_type":             by_type,
            "success_rate":        round(improved / len(outcomes), 2) if outcomes else None,
            "last_intervention":   self.episodes[-1]["intervention_type"],
        }