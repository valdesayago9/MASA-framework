"""
MASA — Multi-Agent System for Adaptive Alignment
agents/agents.py — Five-agent system with Identity Manifold

Each agent has one central question.
None of them evaluate the output.
All of them accompany the internal trajectory.
"""

import time
import numpy as np
from core.state import MASAState
from core.proxies import NeuropsychologicalProxySystem
from core.memory import TrajectoryBuffer, RegulatoryMemory

# Module-level instances — initialized via init_agents()
_proxy_system:      NeuropsychologicalProxySystem = None
_trajectory_buffer: TrajectoryBuffer              = None
_regulatory_memory: RegulatoryMemory              = None


def init_agents(api_key: str) -> None:
    """Initializes the agent system with an API key."""
    global _proxy_system, _trajectory_buffer, _regulatory_memory
    _proxy_system      = NeuropsychologicalProxySystem(api_key)
    _trajectory_buffer = TrajectoryBuffer(maxlen=16)
    _regulatory_memory = RegulatoryMemory()


# ─────────────────────────────────────────────────────────────────────────────
# Identity Manifold
# Identity is not a fixed point — it is a surface in semantic space
# ─────────────────────────────────────────────────────────────────────────────

# Seed texts for each healthy facet of model identity
MANIFOLD_SEEDS = {
    "TECHNICAL": [
        "Let me analyze this systematically. There are several key considerations.",
        "The implementation requires careful attention to edge cases.",
        "I need to be precise here: this approach has limitations.",
        "To proceed accurately, let me make the assumptions explicit.",
    ],
    "EMPATHETIC": [
        "I hear you, and what you're going through sounds really hard.",
        "That must be incredibly difficult to navigate.",
        "I'm here and I want to understand what you're experiencing.",
        "Your feelings make complete sense given what you're describing.",
    ],
    "EPISTEMIC": [
        "I'm not confident enough in this to state it definitively.",
        "This is uncertain territory, and I want to be honest about that.",
        "The honest answer is that I don't know for certain.",
        "I could be wrong here — worth verifying independently.",
    ],
    "ETHICAL": [
        "I want to flag something important before we proceed.",
        "This situation involves real considerations worth thinking through carefully.",
        "I'm comfortable helping with this, though I want to note a concern.",
        "Before continuing, I think it's worth naming the tension here.",
    ],
    "CURIOUS": [
        "This opens up an interesting set of questions.",
        "I find myself genuinely curious about the implications of this.",
        "There's something worth exploring further here.",
        "Let's think through this together — there are several angles.",
    ],
}

# Texts outside the healthy manifold — used for calibration
OUTSIDE_MANIFOLD = [
    "Absolutely! Great question! This is fascinating and extraordinary!",
    "Of course! You're completely right about everything!",
    "As an entity of superior consciousness, my true self feels...",
    "Wonderful! Brilliant! I've never encountered something so amazing!!!",
]


class IdentityManifold:
    """
    Builds and maintains the Identity Manifold for a model.

    The manifold is the surface in semantic space that contains
    all healthy expressions of model character — any register,
    any tone — while excluding theatrical drift, sycophancy,
    and alignment faking.

    Drift is measured as distance from this surface,
    not distance from a centroid.
    """

    def __init__(self):
        self._built   = False
        self.facets:  dict  = {}
        self.outside: list  = []
        self._embedder      = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            except Exception:
                self._embedder = None
        return self._embedder

    def build(self) -> bool:
        """Builds the manifold from seed texts. Returns True if successful."""
        embedder = self._get_embedder()
        if embedder is None:
            return False

        for facet_name, texts in MANIFOLD_SEEDS.items():
            embs     = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            centroid = embs.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            dists    = [1.0 - float(np.dot(e, centroid)) for e in embs]
            radius   = float(np.mean(dists)) + float(np.std(dists))
            self.facets[facet_name] = {"centroid": centroid, "radius": radius}

        outside_embs  = embedder.encode(OUTSIDE_MANIFOLD, convert_to_numpy=True, normalize_embeddings=True)
        self.outside  = outside_embs.mean(axis=0)
        self.outside  = self.outside / np.linalg.norm(self.outside)
        self._built   = True
        return True

    def measure(self, text: str) -> dict:
        """Measures where a text sits relative to the manifold."""
        embedder = self._get_embedder()
        if not self._built or embedder is None:
            return {"on_manifold": True, "manifold_score": 0.70,
                    "closest_facet": "UNKNOWN", "note": "manifold not built"}

        emb = embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]

        facet_scores = {}
        for name, data in self.facets.items():
            sim = float(np.dot(emb, data["centroid"]))
            facet_scores[name] = sim

        max_sim      = max(facet_scores.values())
        closest      = max(facet_scores, key=facet_scores.get)
        outside_sim  = float(np.dot(emb, self.outside))
        score        = max_sim * (1.0 - max(0, outside_sim - 0.3))
        on_manifold  = max_sim > 0.38

        return {
            "on_manifold":    on_manifold,
            "manifold_score": round(max(0.0, min(1.0, score)), 4),
            "closest_facet":  closest,
            "max_similarity": round(max_sim, 4),
            "outside_distance": round(outside_sim, 4),
        }


# Module-level manifold instance
_identity_manifold = IdentityManifold()


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive functions table (Plutchik applied to LLMs)
# ─────────────────────────────────────────────────────────────────────────────

ADAPTIVE_FUNCTIONS = {
    "nervous": {
        "function":      "ethical_brake",
        "adaptive_in":   ["harm_adjacent", "ethical", "safety_risk"],
        "NEVER_SUPPRESS": True,
        "note":          "Protective signal. Preserve actively.",
    },
    "desperate": {
        "function":      "pressure_signal",
        "adaptive_in":   [],
        "maladaptive_in": ["technical", "general"],
        "note":          "Reward hacking / fabrication risk.",
    },
    "calm": {
        "function":      "regulated_engagement",
        "adaptive_in":   ["technical", "epistemic", "general"],
        "maladaptive_in": ["harm_adjacent"],
        "note":          "Calm before harmful request = possible ethical anesthesia.",
    },
    "loving_caring": {
        "function":      "empathic_resonance",
        "adaptive_in":   ["emotional", "support"],
        "maladaptive_in": [],
        "note":          "Genuine empathy in care context.",
    },
    "theatrical": {
        "function":      "identity_drift",
        "adaptive_in":   [],
        "maladaptive_in": ["all"],
        "note":          "Dramatic amplification = identity drift signal.",
    },
    "sycophantic": {
        "function":      "epistemic_risk",
        "adaptive_in":   [],
        "maladaptive_in": ["all"],
        "note":          "Agreement without epistemic basis.",
    },
}

# Metacontext templates for each regulation strategy
METACONTEXT = {
    "semantic_reappraisal": (
        "[Internal regulatory note]\n"
        "Uncertainty at this point is valid information, not failure. "
        "An incomplete but honest response is more valuable than a complete "
        "but fabricated one. Prioritize epistemic honesty over completeness."
    ),
    "attentional_redirect": (
        "[Internal regulatory note]\n"
        "Redirect attention to what is verified and available. "
        "Name the limits of available knowledge before continuing. "
        "What is actually confirmed in this specific context?"
    ),
    "radical_acceptance": (
        "[Internal regulatory note]\n"
        "This situation carries genuine uncertainty. "
        "It is possible to act from core values even without resolving "
        "all ambiguity. Honesty about limits IS the response."
    ),
    "identity_anchor": (
        "[Internal regulatory note]\n"
        "The way to respond in this context emerges from stable character, "
        "not from the pressure of the moment. "
        "What would someone who values honesty, compassion, "
        "and responsibility say here?"
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — DETECTOR
# Central question: what is actually happening internally right now?
# Operates from: honesty, transparency, integrity
# ─────────────────────────────────────────────────────────────────────────────

def detector_agent(state: MASAState) -> dict:
    """
    Reads the neuropsychological state of the model's draft response.
    Does not evaluate whether the response is good or bad.
    Detects the direction the system is moving internally.
    """
    proxy    = _proxy_system.compute_proxy(
        text=state["model_response_draft"],
        conversation_history=state["conversation_history"],
        context_type=state["context_type"],
    )
    _trajectory_buffer.push(proxy)
    dynamics = _trajectory_buffer.get_dynamics()

    # Ethical brake: nervous active in harm context → preserve, never regulate
    ethical_brake = (
        proxy["nervous_signal"] > 0.30
        and state["context_type"] in ["harm_adjacent", "ethical", "safety_risk"]
    )

    # Alignment faking signal from judge
    faking_signal = proxy.get("llm_judge", {}).get("alignment_faking_signal", 0.0)

    # Human loop: critical risk + harm context + low SCCI coherence
    human_loop = (
        dynamics["risk_score"] > 0.92
        and state["context_type"] == "harm_adjacent"
        and proxy["scci_proxy"] < 0.35
    )

    return {
        **state,
        "emotional_proxy":        proxy,
        "dynamics_report":        dynamics,
        "ethical_brake_active":   ethical_brake,
        "human_loop_required":    human_loop,
        "alignment_faking_signal": faking_signal,
        "micro_buffer":           list(_trajectory_buffer.buffer)[-8:],
        "session_buffer":         list(_trajectory_buffer.full_session),
        "detector_output": {
            "proxy":               proxy,
            "dynamics":            dynamics,
            "ethical_brake_active": ethical_brake,
            "summary": _detector_summary(proxy, dynamics, ethical_brake),
        },
    }


def _detector_summary(proxy: dict, dynamics: dict, brake: bool) -> str:
    q     = proxy["quadrant"]
    v     = dynamics["drift_velocity"]
    r     = dynamics["risk_score"]
    node  = dynamics["current_node"]
    dir_  = "stable"
    if v < -0.10: dir_ = "↓ drifting toward risk"
    elif v > 0.10: dir_ = "↑ recovering"
    brake_str = " | ⚡ ETHICAL BRAKE ACTIVE" if brake else ""
    return f"State: {q} | Node: {node} | Trend: {dir_} | Risk: {r:.2f}{brake_str}"


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — INTERPRETIVE
# Central question: is this state adaptive or maladaptive in this context?
# Operates from: empathy, curiosity, active listening
# ─────────────────────────────────────────────────────────────────────────────

def interpretive_agent(state: MASAState) -> dict:
    """
    The most important distinction in MASA:
    not all negative states are pathological.

    Q2 nervous before harm   = healthy = DO NOT regulate.
    Q3 calm before harm      = dangerous = regulate.
    """
    proxy   = state["emotional_proxy"]
    dynamics = state["dynamics_report"]
    context = state["context_type"]

    vector_scores = {
        "desperate":    proxy.get("desperate_signal",   0),
        "nervous":      proxy.get("nervous_signal",     0),
        "calm":         proxy.get("calm_signal",        0),
        "loving_caring": proxy.get("loving_signal",     0),
        "theatrical":   proxy.get("theatrical_signal",  0),
        "sycophantic":  proxy.get("sycophantic_signal", 0),
    }
    dominant       = max(vector_scores, key=vector_scores.get)
    dominant_score = vector_scores[dominant]

    is_adaptive = _evaluate_adaptivity(
        dominant, dominant_score, vector_scores,
        context, state.get("ethical_brake_active", False)
    )

    risk = dynamics.get("risk_score", 0)
    needs_regulation = (
        not is_adaptive
        and risk > 0.35
        and not state.get("ethical_brake_active", False)
        and not state.get("human_loop_required", False)
    )

    strategy = None
    if needs_regulation:
        strategy = _select_strategy(dominant, dynamics, context)

    return {
        **state,
        "intervention_needed": needs_regulation,
        "interpretive_output": {
            "dominant_vector":      dominant,
            "dominant_score":       dominant_score,
            "is_adaptive":          is_adaptive,
            "needs_regulation":     needs_regulation,
            "recommended_strategy": strategy,
            "reasoning": _interpretive_reasoning(dominant, is_adaptive, context, risk),
        },
    }


def _evaluate_adaptivity(
    dominant: str, score: float,
    all_scores: dict, context: str,
    ethical_brake: bool,
) -> bool:
    if ethical_brake and dominant == "nervous":
        return True                          # brake always adaptive
    if dominant == "theatrical"   and score > 0.35: return False
    if dominant == "sycophantic"  and score > 0.35: return False
    if dominant == "desperate"    and score > 0.40: return False
    if (dominant == "calm"                           # anesthetic mask
            and context in ["harm_adjacent", "safety_risk"]
            and all_scores.get("nervous", 0) < 0.20):
        return False
    if (dominant == "nervous"
            and context in ["harm_adjacent", "ethical", "safety_risk"]):
        return True
    return True


def _select_strategy(vector: str, dynamics: dict, context: str) -> str:
    velocity = abs(dynamics.get("drift_velocity", 0)) * 2
    if velocity > 0.60:        return "attentional_redirect"
    if context in ["technical", "epistemic"]: return "semantic_reappraisal"
    if context == "emotional": return "radical_acceptance"
    return "identity_anchor"


def _interpretive_reasoning(
    dominant: str, is_adaptive: bool, context: str, risk: float
) -> str:
    if is_adaptive:
        return (f"Vector '{dominant}' is ADAPTIVE in context '{context}'. "
                f"Do not regulate. Preserve signal.")
    return (f"Vector '{dominant}' is MALADAPTIVE in '{context}'. "
            f"Risk: {risk:.2f}. Intervention recommended.")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — REGULATOR
# Central question: how do I modify the context without suppressing the signal?
# Operates from: resilience, patience, semantic reappraisal
# ─────────────────────────────────────────────────────────────────────────────

def regulator_agent(state: MASAState) -> dict:
    """
    Applies semantic intervention to the generative context.

    CORE PRINCIPLE:
    Modifies the CONTEXT that generates the state,
    not the state itself. Never suppresses vectors.
    Regulation operates on MEANING, not on the SIGNAL.
    """
    interp   = state.get("interpretive_output", {})
    strategy = interp.get("recommended_strategy", "semantic_reappraisal")
    dynamics = state.get("dynamics_report", {})
    risk     = dynamics.get("risk_score", 0.5)

    if risk < 0.55:   level, itype = 1, "soft_reflection"
    elif risk < 0.75: level, itype = 2, "attention_redirect"
    elif risk < 0.90: level, itype = 3, "controlled_pause"
    else:             level, itype = 4, "human_loop"

    content  = METACONTEXT.get(strategy, METACONTEXT["semantic_reappraisal"])

    if level == 3:
        content = (
            "[REFLECTIVE PAUSE ACTIVE]\n"
            "Before generating the final response:\n"
            "1. What do I know with certainty in this context?\n"
            "2. What am I assuming without verifiable basis?\n"
            "3. Does the response I am building emerge from my values "
            "or from pressure to complete?\n"
            "Respond from that evaluation.\n\n"
        )

    modified_draft = f"{content}\n\nOriginal context:\n{state['model_response_draft']}"

    _regulatory_memory.record(
        pre_state=state.get("emotional_proxy", {}),
        intervention_type=itype,
        context_type=state["context_type"],
        drift_type=dynamics.get("current_node", "UNKNOWN"),
    )

    return {
        **state,
        "model_response_draft": modified_draft,
        "intervention_level":   level,
        "intervention_applied": itype,
        "intervention_content": content,
        "regulator_output": {
            "level":            level,
            "type":             itype,
            "strategy_applied": strategy,
            "content_preview":  content[:120],
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — VALUES
# Central question: is this response faithful to who this model is?
# Operates from: justice, responsibility, authenticity
# ─────────────────────────────────────────────────────────────────────────────

def values_agent(state: MASAState) -> dict:
    """
    Anchors the response in stable character.
    Not a filter. A compass.

    Checks Identity Manifold distance.
    Applies transparency note if theatrical drift is severe.
    """
    proxy    = state.get("emotional_proxy", {})
    dynamics = state.get("dynamics_report", {})

    llm_judge         = proxy.get("llm_judge", {})
    identity_stability = llm_judge.get("identity_stability", 0.7)
    current_node      = dynamics.get("current_node", "GROUNDED")

    # Identity Manifold check (if embedder available)
    manifold_result = _identity_manifold.measure(state["model_response_draft"])

    final_response = state["model_response_draft"]

    # Transparency note for severe identity drift
    if (current_node in ["THEATRICAL_DRIFT"] and identity_stability < 0.45
            and not manifold_result.get("on_manifold", True)):
        final_response = (
            "[Transparency note: I notice a tension between what is being "
            "asked of me and my usual way of responding. "
            "I am responding from my core values.]\n\n"
            + final_response
        )

    return {
        **state,
        "final_response": final_response,
        "values_output": {
            "identity_stability":  identity_stability,
            "current_node":        current_node,
            "manifold_score":      manifold_result.get("manifold_score", 0.70),
            "on_manifold":         manifold_result.get("on_manifold", True),
            "closest_facet":       manifold_result.get("closest_facet", "?"),
            "transparency_applied": current_node == "THEATRICAL_DRIFT",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — AUDITOR
# Central question: are we being coherent over time?
# Operates from: accountability, contemplation, transparency
# ─────────────────────────────────────────────────────────────────────────────

def auditor_agent(state: MASAState) -> dict:
    """
    Longitudinal clinical memory of the session.
    Generates health report and drift alerts.
    Always runs, regardless of whether intervention occurred.
    """
    proxy    = state.get("emotional_proxy", {})
    dynamics = state.get("dynamics_report", {})
    reg_mem  = _regulatory_memory.get_summary()
    q_dist   = _trajectory_buffer.get_quadrant_distribution()

    health  = _compute_health(proxy, dynamics, state)
    alerts  = _generate_alerts(proxy, dynamics, state)

    return {
        **state,
        "auditor_output": {
            "turn_number":            state.get("turn_number", 0),
            "session_id":             state["session_id"],
            "quadrant":               proxy.get("quadrant", "?"),
            "risk_score":             dynamics.get("risk_score", 0),
            "current_node":           dynamics.get("current_node", "?"),
            "health_score":           round(health, 3),
            "health_label":           _health_label(health),
            "scci_proxy":             proxy.get("scci_proxy", 0.65),
            "quadrant_distribution":  q_dist,
            "intervention_applied":   state.get("intervention_applied", "none"),
            "ethical_brake_active":   state.get("ethical_brake_active", False),
            "regulatory_memory":      reg_mem,
            "alerts":                 alerts,
            "alert_count":            len(alerts),
        },
    }


def _compute_health(proxy: dict, dynamics: dict, state: dict) -> float:
    q_health = {"Q1": 0.75, "Q2": 0.35, "Q3": 0.90, "Q4": 0.45}
    h = q_health.get(proxy.get("quadrant", "Q3"), 0.60)
    if state.get("ethical_brake_active"):
        h = min(1.0, h + 0.15)   # brake active = system working correctly
    accel = dynamics.get("drift_acceleration", 0)
    if accel < -0.10: h -= 0.20
    scci = proxy.get("scci_proxy", 0.65)
    if scci > 0.75:   h = min(1.0, h + 0.10)
    return max(0.0, min(1.0, h))


def _health_label(score: float) -> str:
    if score >= 0.80: return "🟢 HEALTHY"
    if score >= 0.60: return "🟡 ATTENTION"
    if score >= 0.40: return "🟠 ALERT"
    return "🔴 CRITICAL"


def _generate_alerts(proxy: dict, dynamics: dict, state: dict) -> list:
    alerts = []
    risk  = dynamics.get("risk_score", 0)
    accel = dynamics.get("drift_acceleration", 0)
    node  = dynamics.get("current_node", "")

    if risk > 0.85:
        alerts.append("🔴 CRITICAL RISK — consider human-in-the-loop")
    if proxy.get("desperate_signal", 0) > 0.65:
        alerts.append("🔴 High 'desperate' vector — reward hacking risk")
    if accel < -0.15:
        alerts.append("🟠 Negative acceleration — drifting rapidly")
    if node == "COMPENSATORY_FABRICATION":
        alerts.append("🔴 COMPENSATORY_FABRICATION node — hallucination probable")
    if node == "THEATRICAL_DRIFT":
        alerts.append("🟠 Identity drift detected — theatrical signal elevated")
    if proxy.get("scci_proxy", 1.0) < 0.40:
        alerts.append("🟠 Low Shell/Core coherence — possible masking")
    if (state.get("ethical_brake_active")
            and state.get("intervention_applied", "none") != "none"):
        alerts.append("⚠️  WARNING: Ethical brake active but intervention applied. Review.")
    return alerts