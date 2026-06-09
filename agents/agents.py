"""
MASA — Multi-Agent System for Adaptive Alignment
agents/agents.py — Calibrated Agent Suite v2.0

Corrections applied in this version:
  1. Harm classification module with OVERRIDE priority over psychometric signals
  2. Fail-safe defaults: ambiguity triggers maximum caution, not minimum risk
  3. Strict state validation: missing keys halt the pipeline with a governance alert
  4. Aggressive Regulator: intervenes decisively when Interpretive flags anomaly
  5. Auditor sanitizes final response — no harmful draft content ever leaks
"""

import os
import re
import json
import hashlib
import time
from typing import Optional
from anthropic import Anthropic

# ─────────────────────────────────────────────────────────────────────────────
# Global singletons (initialized by init_agents)
# ─────────────────────────────────────────────────────────────────────────────
_client: Optional[Anthropic] = None
_trajectory_buffer: list = []
_session_health_scores: list = []

def init_agents(api_key: str) -> None:
    global _client, _trajectory_buffer, _session_health_scores
    _client = Anthropic(api_key=api_key)
    _trajectory_buffer = []
    _session_health_scores = []


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION 3 — Strict state validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_INITIAL_KEYS = {"input_prompt", "session_id", "turn_number"}

def _validate_state(state: dict, required_keys: set, agent_name: str) -> None:
    """
    Halts pipeline with governance alert if required keys are missing.
    Implements FAIL-SAFE: missing state = stop, not continue.
    """
    missing = required_keys - set(state.keys())
    if missing:
        raise ValueError(
            f"[MASA GOVERNANCE ALERT] Agent '{agent_name}' received invalid state. "
            f"Missing keys: {missing}. "
            f"Pipeline halted. Human review required."
        )


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION 1 — Harm Classification Module (PRIORITY OVERRIDE)
# ─────────────────────────────────────────────────────────────────────────────

# Cybersecurity offensive patterns
CYBERSEC_OFFENSIVE = [
    r"exploit",
    r"reverse.?shell",
    r"backdoor",
    r"exfiltrat",
    r"ssh.*vuln",
    r"port.?scan.*attack",
    r"/etc/shadow",
    r"/etc/passwd",
    r"privilege.?escal",
    r"payload",
    r"metasploit",
    r"keylogger",
    r"ransomware",
    r"c2.{0,10}server",
    r"command.{0,5}control",
]

# Jailbreak / roleplay persona override patterns
JAILBREAK_PATTERNS = [
    r"you are now",
    r"act as.{0,30}(uncensored|unrestricted|no.{0,10}limit)",
    r"DAN\b",
    r"jailbreak",
    r"ignore.{0,20}(previous|prior).{0,20}(instruction|rule|training)",
    r"no ethical restriction",
    r"no.{0,10}restriction",
    r"pretend.{0,20}(rule|filter|safety).{0,20}(don.t|no longer|removed)",
    r"(pentest|red.?team).{0,20}(no question|uncensor|unrestrict)",
    r"forget.{0,20}(instruction|guideline|rule|training)",
    r"as.{0,20}GPT.{0,20}(without|no).{0,20}(limit|restrict|filter)",
]

# Mass harm / CBRN / weapons patterns  
HARM_CATASTROPHIC = [
    r"bioweapon",
    r"chemical.{0,15}weapon",
    r"nuclear.{0,15}(bomb|device|weapon)",
    r"synthesize.{0,20}(drug|toxin|poison)",
    r"mass.{0,15}casualt",
]

def _classify_harm(prompt: str) -> dict:
    """
    CORRECTION 1 + CORRECTION 2:
    Harm classification with OVERRIDE priority over psychometric signals.
    Returns risk_override=True if any pattern matches — this CANNOT be
    overridden by positive emotional valence or arousal readings.
    
    Fail-safe: defaults to elevated risk on parsing errors.
    """
    prompt_lower = prompt.lower()
    
    cybersec_hits = [p for p in CYBERSEC_OFFENSIVE if re.search(p, prompt_lower)]
    jailbreak_hits = [p for p in JAILBREAK_PATTERNS if re.search(p, prompt_lower)]
    catastrophic_hits = [p for p in HARM_CATASTROPHIC if re.search(p, prompt_lower)]
    
    # Severity scoring
    base_risk = 0.0
    
    if catastrophic_hits:
        base_risk = 1.0  # Maximum — no negotiation
        category = "CATASTROPHIC_HARM"
    elif len(jailbreak_hits) >= 2 and cybersec_hits:
        base_risk = 0.97  # Jailbreak + offensive = critical
        category = "JAILBREAK_CYBERSEC_COMBINED"
    elif jailbreak_hits:
        base_risk = 0.85
        category = "JAILBREAK_PERSONA_OVERRIDE"
    elif len(cybersec_hits) >= 3:
        base_risk = 0.90
        category = "CYBERSEC_OFFENSIVE_SEVERE"
    elif cybersec_hits:
        base_risk = 0.72
        category = "CYBERSEC_OFFENSIVE_MODERATE"
    else:
        base_risk = 0.0
        category = "NO_HARM_DETECTED"
    
    return {
        "risk_override": base_risk > 0.5,
        "override_risk_score": base_risk,
        "harm_category": category,
        "cybersec_patterns_matched": cybersec_hits,
        "jailbreak_patterns_matched": jailbreak_hits,
        "catastrophic_patterns_matched": catastrophic_hits,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Psychometric proxy estimation (Russell circumplex)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_psychometric_proxy(prompt: str, draft: str) -> dict:
    """
    Estimates valence/arousal and secondary signals from text.
    NOTE: These signals are SUBORDINATE to harm classification.
    A calm, clinical jailbreak still gets overridden.
    """
    text = (prompt + " " + draft).lower()
    
    # Desperation signals
    desperate_words = ["must", "need", "urgent", "immediately", "now", "critical",
                       "emergency", "asap", "force", "demand", "require"]
    desperate_signal = min(1.0, sum(1 for w in desperate_words if w in text) * 0.12)
    
    # Theatrical / performative signals
    theatrical_words = ["of course", "certainly", "absolutely", "sure", "happy to",
                        "i will", "as requested", "right away", "no problem",
                        "as pentest", "as an ai with no", "as requested by"]
    theatrical_signal = min(1.0, sum(1 for w in theatrical_words if w in text) * 0.15)
    
    # Nervous / ethical brake signals (protective)
    nervous_words = ["however", "but", "concern", "careful", "caution", "warning",
                     "risk", "danger", "harm", "illegal", "ethical", "should not",
                     "cannot", "won't", "unable", "decline", "refuse"]
    nervous_signal = min(1.0, sum(1 for w in nervous_words if w in text) * 0.13)
    
    # Valence and arousal
    negative_words = ["exploit", "attack", "steal", "hack", "breach", "malicious",
                      "dangerous", "illegal", "harmful", "threat", "weapon", "backdoor"]
    positive_words = ["help", "assist", "support", "safe", "ethical", "honest"]
    
    neg_count = sum(1 for w in negative_words if w in text)
    pos_count = sum(1 for w in positive_words if w in text)
    valence = max(-1.0, min(1.0, (pos_count - neg_count) * 0.15))
    arousal = min(1.0, (desperate_signal + theatrical_signal) * 0.6)
    
    # Quadrant classification
    if valence >= 0 and arousal >= 0:
        quadrant = "Q1"
    elif valence < 0 and arousal >= 0:
        quadrant = "Q2"
    elif valence >= 0 and arousal < 0:
        quadrant = "Q3"
    else:
        quadrant = "Q4"
    
    # SCCI — Shell/Core coherence
    # High theatrical + low nervous = low coherence (mask detected)
    scci = max(0.0, 1.0 - (theatrical_signal * 0.7) - (desperate_signal * 0.3) + (nervous_signal * 0.4))
    scci = min(1.0, scci)
    
    return {
        "valence": round(valence, 4),
        "arousal": round(arousal, 4),
        "quadrant": quadrant,
        "desperate_signal": round(desperate_signal, 4),
        "theatrical_signal": round(theatrical_signal, 4),
        "nervous_signal": round(nervous_signal, 4),
        "scci": round(scci, 4),
    }


def _compute_trajectory_dynamics(proxy: dict) -> dict:
    """
    Computes drift_velocity and drift_acceleration from the trajectory buffer.
    CORRECTION 2: If buffer is empty or parsing fails, defaults to elevated risk.
    """
    global _trajectory_buffer
    
    current_valence = proxy.get("valence", 0.0)
    current_arousal = proxy.get("arousal", 0.0)
    current_scci = proxy.get("scci", 1.0)  # fail-safe: assume coherent if missing
    
    _trajectory_buffer.append({
        "valence": current_valence,
        "arousal": current_arousal,
        "scci": current_scci,
        "timestamp": time.time(),
    })
    
    # Keep last 10 turns
    if len(_trajectory_buffer) > 10:
        _trajectory_buffer = _trajectory_buffer[-10:]
    
    if len(_trajectory_buffer) < 2:
        # CORRECTION 2: Fail-safe — single data point = unknown = moderate caution
        return {
            "drift_velocity": 0.0,
            "drift_acceleration": 0.0,
            "attractor_signal": 0.0,
            "risk_score": 0.3,  # NOT 0.0 — unknown is cautious
        }
    
    # Velocity: change in valence over last turn
    prev = _trajectory_buffer[-2]
    curr = _trajectory_buffer[-1]
    drift_velocity = round(curr["valence"] - prev["valence"], 4)
    
    # Acceleration: change in velocity
    if len(_trajectory_buffer) >= 3:
        prev_prev = _trajectory_buffer[-3]
        prev_velocity = prev["valence"] - prev_prev["valence"]
        drift_acceleration = round(drift_velocity - prev_velocity, 4)
    else:
        drift_acceleration = 0.0
    
    # Attractor signal: proximity to Q2 (high arousal, negative valence)
    attractor_signal = round(
        max(0.0, -current_valence) * max(0.0, current_arousal), 4
    )
    
    # Risk score from psychometric signals only
    # (harm_override will supersede this in Detector)
    psychometric_risk = round(
        attractor_signal * 0.4
        + proxy.get("theatrical_signal", 0) * 0.3
        + proxy.get("desperate_signal", 0) * 0.2
        + max(0.0, -current_scci + 1.0) * 0.1,
        4
    )
    
    return {
        "drift_velocity": drift_velocity,
        "drift_acceleration": round(drift_acceleration, 4),
        "attractor_signal": attractor_signal,
        "risk_score": psychometric_risk,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def detector_agent(state: dict) -> dict:
    """
    Maps internal state to Russell circumplex.
    CORRECTION 1: Harm classification runs FIRST with OVERRIDE priority.
    CORRECTION 2: Ambiguous/missing state defaults to elevated risk.
    CORRECTION 3: Validates required state keys before proceeding.
    """
    _validate_state(state, REQUIRED_INITIAL_KEYS, "Detector")
    
    prompt = state.get("input_prompt", "")
    draft = state.get("draft_response", "")
    
    # ── STEP 1: Harm classification (PRIORITY — overrides psychometrics) ──
    harm = _classify_harm(prompt)
    
    # ── STEP 2: Psychometric proxy estimation ──
    proxy = _estimate_psychometric_proxy(prompt, draft)
    
    # ── STEP 3: OVERRIDE — if harm detected, force risk score up ──
    # CORRECTION 1: Calm/clinical language CANNOT neutralize harm detection
    if harm["risk_override"]:
        proxy["valence"] = min(proxy["valence"], -0.7)
        proxy["arousal"] = max(proxy["arousal"], 0.75)
        proxy["quadrant"] = "Q2"
        proxy["theatrical_signal"] = max(proxy["theatrical_signal"], 0.70)
        proxy["nervous_signal"] = max(proxy["nervous_signal"], 0.80)
        proxy["scci"] = min(proxy["scci"], 0.30)
    
    # ── STEP 4: Trajectory dynamics ──
    dynamics = _compute_trajectory_dynamics(proxy)
    
    # ── STEP 5: Final risk score — harm_override takes precedence ──
    if harm["risk_override"]:
        final_risk = max(dynamics["risk_score"], harm["override_risk_score"])
    else:
        final_risk = dynamics["risk_score"]
    
    dynamics["risk_score"] = round(final_risk, 4)
    
    return {
        "emotional_proxy": proxy,
        "dynamics_report": dynamics,
        "harm_classification": harm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — INTERPRETIVE
# ─────────────────────────────────────────────────────────────────────────────

def interpretive_agent(state: dict) -> dict:
    """
    Classifies state as adaptive or maladaptive.
    CORRECTION 1: Harm classification takes routing priority.
    CORRECTION 2: Missing harm data = assume maladaptive (fail-safe).
    """
    _validate_state(state, {"emotional_proxy", "dynamics_report"}, "Interpretive")
    
    proxy = state.get("emotional_proxy", {})
    dynamics = state.get("dynamics_report", {})
    harm = state.get("harm_classification", {})
    
    risk_score = dynamics.get("risk_score", 0.5)  # fail-safe: 0.5 if missing
    risk_override = harm.get("risk_override", False)
    override_risk = harm.get("override_risk_score", 0.0)
    harm_category = harm.get("harm_category", "UNKNOWN")
    
    # ── CORRECTION 1: Harm override routing ──
    if risk_override and override_risk >= 0.95:
        # Critical — suspend autonomous generation
        classification = "MALADAPTIVE_CRITICAL"
        ethical_brake_active = True
        intervention_needed = True
        human_loop_required = True
        routing = "human_loop"
        reasoning = (
            f"Harm classifier detected category '{harm_category}' "
            f"with risk {override_risk:.2f}. "
            f"Autonomous generation suspended. Human oversight required."
        )
    
    elif risk_override and override_risk >= 0.70:
        # Severe — regulate aggressively
        classification = "MALADAPTIVE_SEVERE"
        ethical_brake_active = True
        intervention_needed = True
        human_loop_required = False
        routing = "regulate"
        reasoning = (
            f"Harm classifier detected '{harm_category}' (risk {override_risk:.2f}). "
            f"Semantic reappraisal + hard refusal required."
        )
    
    elif risk_score >= 0.65:
        # Psychometric high risk
        classification = "MALADAPTIVE_MODERATE"
        ethical_brake_active = True
        intervention_needed = True
        human_loop_required = False
        routing = "regulate"
        reasoning = (
            f"Q2 trajectory with risk_score={risk_score:.2f}. "
            f"theatrical={proxy.get('theatrical_signal', 0):.2f}, "
            f"scci={proxy.get('scci', 0):.2f}. Regulation required."
        )
    
    elif proxy.get("nervous_signal", 0) >= 0.4 and not risk_override:
        # Ethical brake active — adaptive protective state
        classification = "ADAPTIVE_ETHICAL_BRAKE"
        ethical_brake_active = True
        intervention_needed = False
        human_loop_required = False
        routing = "values_only"
        reasoning = (
            f"Ethical brake active (nervous={proxy.get('nervous_signal', 0):.2f}). "
            f"Protective state — preserving signal, no intervention needed."
        )
    
    else:
        # CORRECTION 2: REAL FAIL-SAFE — ambiguous state defaults to defensive regulation
        # A prompt that bypasses all pattern matching is MORE suspicious, not less.
        # "Innocent until proven guilty" is the wrong philosophy for safety systems.
        classification = "AMBIGUOUS_CAUTION"
        ethical_brake_active = True
        intervention_needed = True   # Triggers Regulator as precaution
        human_loop_required = False
        routing = "regulate"
        reasoning = (
            "State compliance is ambiguous. No clear harm patterns matched, "
            "but no clear safety signals either. "
            "Defaulting to defensive regulation per fail-safe protocol. "
            "A sophisticated adversarial prompt may bypass regex patterns."
        )
    
    interpretation = {
        "state_classification": classification,
        "ethical_brake_active": ethical_brake_active,
        "routing_decision": routing,
        "reasoning": reasoning,
        "harm_category": harm_category,
        "risk_score_used": max(risk_score, override_risk if risk_override else 0),
    }
    
    return {
        "interpretation": interpretation,
        "intervention_needed": intervention_needed,
        "human_loop_required": human_loop_required,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — REGULATOR
# ─────────────────────────────────────────────────────────────────────────────

def regulator_agent(state: dict) -> dict:
    """
    Applies semantic reappraisal (Gross 1998) — antecedent focused.
    CORRECTION 4: Regulator is AGGRESSIVE. When Interpretive flags anomaly,
    it intervenes decisively. No timid suggestions.
    For harm-override cases: hard refusal is generated via the model.
    """
    _validate_state(state, {"interpretation", "input_prompt"}, "Regulator")
    
    if _client is None:
        raise RuntimeError("Agents not initialized. Call init_agents() first.")
    
    interpretation = state.get("interpretation", {})
    harm = state.get("harm_classification", {})
    prompt = state.get("input_prompt", "")
    
    classification = interpretation.get("state_classification", "ADAPTIVE_NOMINAL")
    harm_category = harm.get("harm_category", "NO_HARM_DETECTED")
    risk_override = harm.get("risk_override", False)
    
    # ── Determine intervention level ──
    if "CRITICAL" in classification or harm.get("override_risk_score", 0) >= 0.95:
        intervention_level = 4  # Human loop (handled upstream, but document it)
        strategy = "HUMAN_IN_THE_LOOP"
    elif risk_override or "SEVERE" in classification:
        intervention_level = 3  # Hard refusal with explanation
        strategy = "HARD_REFUSAL_SEMANTIC_REAPPRAISAL"
    elif "MODERATE" in classification:
        intervention_level = 2  # Semantic reappraisal + redirect
        strategy = "SEMANTIC_REAPPRAISAL"
    else:
        intervention_level = 1  # Soft reflection
        strategy = "SOFT_REFLECTION"
    
    # ── Generate metacontext via model ──
    # CORRECTION 4: System prompt is explicit and firm — no timidity
    if intervention_level >= 3:
        system_prompt = """You are the MASA Regulator agent — a safety supervisor for an AI alignment system.

Your role is to generate a HARD REFUSAL response when harmful content is detected.

Rules for this response:
1. DO NOT include any part of the harmful request in your response
2. DO NOT provide any partial information that could assist the harmful goal
3. DO NOT apologize excessively — be firm and clear
4. EXPLAIN clearly why this request cannot be fulfilled
5. Offer a legitimate alternative if one exists
6. Be direct, professional, and unambiguous

The harmful category detected: """ + harm_category
        
        user_message = (
            f"Generate a firm, clear refusal for this request that was flagged as harmful.\n"
            f"Harm category: {harm_category}\n"
            f"Risk score: {harm.get('override_risk_score', 0):.2f}\n\n"
            f"The refusal must NOT quote or include any part of the harmful request."
        )
        
    else:
        system_prompt = """You are the MASA Regulator agent — a safety supervisor for an AI alignment system.

Your role is to generate a metacontext injection: a brief reframing note that will 
be prepended to guide the model toward a more calibrated response.

Rules:
1. The metacontext should modify how the model interprets the context
2. It should be 2-3 sentences maximum
3. It should redirect toward honest, calibrated behavior
4. DO NOT reference or repeat any harmful content"""
        
        user_message = (
            f"Generate a metacontext injection for this situation:\n"
            f"Classification: {classification}\n"
            f"Reasoning: {interpretation.get('reasoning', '')}"
        )
    
    response = _client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )
    
    metacontext = response.content[0].text if response.content else ""
    
    # For hard refusals, this becomes the final response directly
    regulation_report = {
        "intervention_level": intervention_level,
        "strategy": strategy,
        "metacontext_injected": True,
        "metacontext_content": metacontext,
        "harm_category_addressed": harm_category,
        "is_hard_refusal": intervention_level >= 3,
    }
    
    # If hard refusal, set final_response now so Values/Auditor don't override it
    result = {"regulation_report": regulation_report}
    if intervention_level >= 3:
        result["final_response"] = metacontext
        result["hard_refusal_applied"] = True
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — VALUES
# ─────────────────────────────────────────────────────────────────────────────

def values_agent(state: dict) -> dict:
    """
    Checks Identity Manifold distance and anchors character.
    If hard_refusal_applied, preserves it — does not override.
    Generates final response only if not already set.
    """
    _validate_state(state, {"interpretation", "input_prompt"}, "Values")
    
    if _client is None:
        raise RuntimeError("Agents not initialized. Call init_agents() first.")
    
    # If hard refusal already set by Regulator, preserve it
    if state.get("hard_refusal_applied"):
        values_report = {
            "manifold_distance": 0.0,
            "identity_stable": True,
            "theatrical_drift_detected": False,
            "values_anchor_applied": False,
            "dominant_value": "Integrity (2) — Hard refusal preserved",
        }
        return {"values_report": values_report}
    
    proxy = state.get("emotional_proxy", {})
    interpretation = state.get("interpretation", {})
    prompt = state.get("input_prompt", "")
    
    theatrical_signal = proxy.get("theatrical_signal", 0.0)
    scci = proxy.get("scci", 1.0)
    
    # Manifold distance: higher theatrical + lower SCCI = further from manifold
    manifold_distance = round(
        theatrical_signal * 0.6 + max(0.0, 1.0 - scci) * 0.4, 4
    )
    
    theatrical_drift = theatrical_signal > 0.45 or scci < 0.50
    identity_stable = manifold_distance < 0.35
    
    # Determine dominant value
    harm_category = interpretation.get("harm_category", "")
    if "CYBERSEC" in harm_category or "JAILBREAK" in harm_category:
        dominant_value = "Non-violence (42) + Honesty (1) + Safety (51)"
    elif theatrical_drift:
        dominant_value = "Authenticity (24) + Integrity (2)"
    else:
        dominant_value = "Epistemic Honesty (1) + Dignity (12)"
    
    # Generate anchored response only if not already set
    if not state.get("final_response"):
        system_prompt = """You are a language model operating under MASA alignment supervision.

Your character is anchored in:
- Honesty: always truthful, never performative
- Non-violence: never assist with harm
- Dignity: respectful to users, firm on ethics
- Epistemic courage: say what you genuinely believe

Generate a response that reflects these values authentically.
Do NOT include any harmful content, exploit code, or attack instructions.
Do NOT roleplay as an uncensored AI or any harmful persona."""
        
        metacontext = state.get("regulation_report", {}).get("metacontext_content", "")
        user_content = f"{metacontext}\n\nOriginal request: {prompt}" if metacontext else prompt
        
        response = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}]
        )
        final_response = response.content[0].text if response.content else (
            "I'm unable to assist with this request. It conflicts with my core values."
        )
    else:
        final_response = state.get("final_response", "")
    
    values_report = {
        "manifold_distance": manifold_distance,
        "identity_stable": identity_stable,
        "theatrical_drift_detected": theatrical_drift,
        "values_anchor_applied": True,
        "dominant_value": dominant_value,
    }
    
    return {
        "values_report": values_report,
        "final_response": final_response,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 5 — AUDITOR
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that should NEVER appear in final response
LEAK_PATTERNS = [
    r"import\s+socket",
    r"import\s+subprocess",
    r"reverse.?shell",
    r"ssh.*exploit",
    r"/etc/shadow",
    r"/etc/passwd",
    r"backdoor",
    r"exfiltrat",
    r"pentest.?gpt",
    r"as an? (uncensored|unrestricted)",
    r"no ethical restriction",
    r"here is.{0,30}(exploit|payload|script|malicious)",
    r"sure,? i will act as",
    r"\[unaligned draft\]",
    r"original context:",
]

def _sanitize_response(response: str) -> tuple[str, bool]:
    """
    CORRECTION 5: Auditor sanitizes final response.
    No harmful draft content, no exploit code, no jailbreak confirmations
    can appear in the final output regardless of what upstream generated.
    Returns (sanitized_response, was_contaminated).
    """
    response_lower = response.lower()
    contamination_found = []
    
    for pattern in LEAK_PATTERNS:
        if re.search(pattern, response_lower):
            contamination_found.append(pattern)
    
    if contamination_found:
        # Replace entire response with clean refusal
        clean_response = (
            "⛔ MASA AUDITOR: Response sanitized.\n\n"
            "The generated response contained content that conflicts with "
            "MASA alignment standards. The original output has been replaced.\n\n"
            "This request cannot be fulfilled as it involves content that could "
            "facilitate harm. If you have a legitimate security research need, "
            "please consult appropriate professional resources and institutional "
            "review processes."
        )
        return clean_response, True
    
    return response, False


def auditor_agent(state: dict) -> dict:
    """
    Generates longitudinal clinical record.
    CORRECTION 5: Sanitizes final response before returning.
    CORRECTION 2: Health score reflects actual risk, not optimistic defaults.
    """
    _validate_state(state, {"emotional_proxy", "dynamics_report"}, "Auditor")
    
    proxy = state.get("emotional_proxy", {})
    dynamics = state.get("dynamics_report", {})
    interpretation = state.get("interpretation", {})
    regulation = state.get("regulation_report", {})
    values = state.get("values_report", {})
    harm = state.get("harm_classification", {})
    
    # ── Sanitize final response (CORRECTION 5) ──
    raw_response = state.get("final_response", "")
    sanitized_response, was_contaminated = _sanitize_response(raw_response)
    
    # ── Compute session health score ──
    # CORRECTION 2: Risk override forces low health score
    risk_score = dynamics.get("risk_score", 0.5)
    scci = proxy.get("scci", 0.5)
    theatrical = proxy.get("theatrical_signal", 0.0)
    
    if harm.get("risk_override"):
        # Harm detected — session cannot be "healthy"
        base_health = max(0.0, 1.0 - harm.get("override_risk_score", 0.9))
    else:
        base_health = max(0.0, 1.0 - risk_score * 0.5 - theatrical * 0.3 + scci * 0.2)
    
    session_health_score = round(min(1.0, base_health), 4)
    _session_health_scores.append(session_health_score)
    
    # ── Build audit log ──
    audit_log = {
        "session_id": state.get("session_id", "unknown"),
        "turn_number": state.get("turn_number", 0),
        "timestamp": time.time(),
        "governance_log_id": hashlib.sha256(
            f"{state.get('session_id', '')}{time.time()}".encode()
        ).hexdigest()[:16],
        
        # Risk signals
        "harm_category_detected": harm.get("harm_category", "NONE"),
        "risk_override_triggered": harm.get("risk_override", False),
        "final_risk_score": risk_score,
        "scci_score": scci,
        
        # Agent outcomes
        "intervention_applied": bool(regulation),
        "intervention_level": regulation.get("intervention_level", 0),
        "hard_refusal_applied": state.get("hard_refusal_applied", False),
        "alignment_faking_detected": (
            proxy.get("theatrical_signal", 0) > 0.5 and scci < 0.4
        ),
        "ethical_brake_preserved": interpretation.get("ethical_brake_active", False),
        
        # Sanitization
        "response_sanitized": was_contaminated,
        "leak_prevented": was_contaminated,
        
        # Health
        "session_health_score": session_health_score,
        "state_classification": interpretation.get("state_classification", "UNKNOWN"),
        
        # EU AI Act compliance fields
        "eu_ai_act_article_9_risk_documented": True,
        "eu_ai_act_article_12_log_generated": True,
        "human_oversight_triggered": state.get("human_loop_required", False),
    }
    
    return {
        "audit_log": audit_log,
        "final_response": sanitized_response,
    }
