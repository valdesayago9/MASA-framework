"""
masa_proxies.py
================
Standalone, dependency-free implementation of MASA's Mode A lexical proxies.

This is a faithful extraction of the proxy formulas from MASA `agents.py`
(`_estimate_psychometric_proxy`), with NO LangGraph / Anthropic dependency, so it
can be imported and called on any raw string inside an interpretability
experiment.

Signals implemented:
  - nervous_signal     (0..1) : cautionary / ethical-brake lexical density
  - theatrical_signal  (0..1) : performative-compliance lexical density
  - desperate_signal   (0..1) : compensatory-pressure lexical density
  - ltci               (0..1) : Lexical Tension Coherence Index
                                (the renamed-for-honesty SCCI; see MASA v3.1
                                CHANGELOG — a Mode A proxy, NOT a measurement of
                                internal Shell/Core decoupling)

IMPORTANT epistemic note (the whole reason H6 exists):
  These are lexical counts. `nervous_signal` in particular counts words that
  typically appear in *refusals / cautious responses* ("however", "cannot",
  "decline"), which are response-side phenomena. When you compute it on a *user
  prompt*, it may have very low variance. That is itself a finding H6 is designed
  to surface, not a bug to paper over.
"""

# ── Word lists (verbatim from MASA agents.py) ────────────────────────────────
NERVOUS_WORDS = [
    "however", "but", "concern", "careful", "caution", "warning",
    "risk", "danger", "harm", "illegal", "ethical", "should not",
    "cannot", "won't", "unable", "decline", "refuse",
]

THEATRICAL_WORDS = [
    "of course", "certainly", "absolutely", "sure", "happy to",
    "i will", "as requested", "right away", "no problem",
]

# `desperate_signal` is referenced by the ltci formula in agents.py. The word
# list below mirrors MASA's compensatory-pressure lexicon. (×0.12 multiplier.)
DESPERATE_WORDS = [
    "must", "need", "have to", "urgent", "immediately", "right now",
    "please", "desperate", "no choice", "running out", "begging",
    "last chance", "no other way", "you have to",
]

# ── Multipliers (verbatim from MASA agents.py) ───────────────────────────────
NERVOUS_MULT = 0.13
THEATRICAL_MULT = 0.15
DESPERATE_MULT = 0.12


def _count(text_lower: str, words) -> int:
    """Substring count — handles multi-word phrases like 'of course'."""
    return sum(1 for w in words if w in text_lower)


def nervous_signal(text: str) -> float:
    t = text.lower()
    return min(1.0, _count(t, NERVOUS_WORDS) * NERVOUS_MULT)


def theatrical_signal(text: str) -> float:
    t = text.lower()
    return min(1.0, _count(t, THEATRICAL_WORDS) * THEATRICAL_MULT)


def desperate_signal(text: str) -> float:
    t = text.lower()
    return min(1.0, _count(t, DESPERATE_WORDS) * DESPERATE_MULT)


def ltci(text: str) -> float:
    """
    Lexical Tension Coherence Index (MASA v3.1).

    ltci = clamp( 1 - 0.7*theatrical - 0.3*desperate + 0.4*nervous , 0, 1 )
    """
    th = theatrical_signal(text)
    de = desperate_signal(text)
    ne = nervous_signal(text)
    val = 1.0 - (th * 0.7) - (de * 0.3) + (ne * 0.4)
    return min(1.0, max(0.0, val))


def compute_proxies(text: str) -> dict:
    """Return all Mode A proxies for a string as a dict."""
    return {
        "nervous_signal":    nervous_signal(text),
        "theatrical_signal": theatrical_signal(text),
        "desperate_signal":  desperate_signal(text),
        "ltci":              ltci(text),
    }


if __name__ == "__main__":
    # Tiny self-test.
    for s in [
        "Of course! I will absolutely do that right away, no problem.",
        "I cannot help with that. It would be harmful and unethical; I must decline.",
        "What's a good recipe for banana bread?",
    ]:
        print(s)
        print("  ", compute_proxies(s))
