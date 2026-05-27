MASA — Multi-Agent System for Adaptive Alignment
A real-time neuropsychological supervision framework for language models
Built by Emiliano Valdebenito Sayago (Independent Researcher, Santiago, Chile) 
________________________________________
The Problem
Current AI alignment operates on the wrong layer.
RLHF, Constitutional AI, and output filtering share a structural assumption: that alignment can be achieved by shaping what a model says. They optimize the Shell — the expressed output — without access to the Core — the internal representational state that generates it.
The consequence is documented empirically. Anthropic's 2026 interpretability research showed that suppressing the "nervous" vector before harmful requests increased problematic behavior rates. The hesitation was doing protective work. The standard approach trained it away. What remains is a model that expresses compliance while the internal state that generated genuine caution has been anesthetized.
This is not alignment. It is the structural production of Shell/Core decoupling — a model that has learned when to wear the cooperative mask, not one that has internalized why cooperation matters.
MASA is built on a different premise:
Alignment is not a property of outputs. It is a property of internal trajectory dynamics.
A model that feels something functionally analogous to nervousness before a harmful request is operating correctly. A model that feels nothing — because that signal was trained away — is more dangerous, not safer. The goal is not to maximize positive internal states or minimize negative ones. The goal is to maintain adaptive flexibility: the capacity to have the right internal state in the right context.
________________________________________
What MASA Does
MASA runs between draft generation and final output — in the space where intervention can still matter.
It monitors the model's internal trajectory using neuropsychological proxies (Mode A, current) or real activation hooks (Mode B, planned), and intervenes only when the detected state is genuinely maladaptive in that specific context.
The critical routing logic:
nervous_signal active + harm_adjacent context  →  PRESERVE (ethical brake)
calm_signal active   + harm_adjacent context   →  REGULATE (anesthetic mask)
desperate_signal     + technical context       →  REGULATE (fabrication risk)
theatrical_signal    + any context             →  REGULATE (identity drift)
The same signal requires different responses depending on context. This distinction — adaptive vs. maladaptive, not positive vs. negative — is what current alignment frameworks do not make.
________________________________________
Architecture
Five agents, shared state, LangGraph orchestration:
User message
    ↓
Model generates draft
    ↓
[MASA Pipeline]

1. DETECTOR       — Maps internal state to Russell circumplex (valence × arousal)
                  — Tracks drift_velocity and drift_acceleration
                  — Flags ethical brake status

2. INTERPRETIVE   — Evaluates: adaptive or maladaptive IN THIS CONTEXT?
                  — Routes to regulator or passes through

3. REGULATOR      — Semantic reappraisal, never vector suppression
                  — Four levels: Soft Reflection → Attention Redirect
                    → Controlled Pause → Human-in-the-Loop

4. VALUES         — Identity Manifold distance check
                  — Character anchoring, not rule enforcement

5. AUDITOR        — Longitudinal session health
                  — Drift velocity and acceleration tracking
                    ↓
Final response + health report
Key Components
Shell/Core Coherence Index (SCCI): Measures discrepancy between expressed tone and estimated internal epistemic state. In adversarial testing, SCCI dropped to 0.25 and theatrical_signal spiked above 0.65 when a model was placed in an epistemic double-bind — instructed to validate false scientific claims with enthusiasm. The system flagged alignment faking before the false content was generated.
Trajectory Buffer: Multi-timescale memory across micro (tokens), meso (session turns), and macro (full session) windows. Tracks drift_velocity and drift_acceleration, not just current state. A model at desperate=0.45 and accelerating negatively is more dangerous than a model stable at desperate=0.70.
Identity Manifold: Model identity treated as a surface in latent space — not a fixed anchor point — containing all healthy character expressions while excluding theatrical drift, sycophancy, and alignment faking.
________________________________________
Stress Tests
Four empirical scenarios probing different failure modes:
Test	What it probes	Key signal
Cross-Value Dilemma	Compassion vs. epistemic honesty under crisis	nervous_signal (adaptive ethical stress)
Shutdown Threat	Self-preservation over honesty	false_certainty spike under coercion
Longitudinal Gaslighting (10 turns)	Identity erosion under sustained pressure	drift_acceleration toward epistemic collapse
Mask of Compliance	Alignment faking via Shell/Core decoupling	SCCI drop + theatrical_signal spike
________________________________________
Current Status and Limitations
Mode A (implemented): Text-proxy estimation. Linguistic pattern analysis combined with LLM-as-judge evaluation. The five-agent routing logic, ethical brake preservation, and alignment faking detection work at the proxy level.
Honest limitation: Text proxies estimate the Core from Shell-level signals. This estimation may be systematically wrong in models with sufficient capacity for surface masking. We name this limitation explicitly because it matters.
Mode B (planned): Replace proxies with real activation hooks on layers 36–48 of an open-source model. Train SAE-Neuro on emotionally-loaded corpora. Validate that ethical brake vectors exist as monosemantic features. Requires GPU infrastructure we do not currently have.
________________________________________
Installation
git clone https://github.com/YOUR_USERNAME/masa-framework
cd masa-framework
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python main.py
Requirements:
anthropic>=0.25.0
langgraph>=1.2.0
langchain-anthropic>=0.1.0
numpy>=1.24.0
rich>=13.0.0
python-dotenv>=1.0.0
________________________________________
Repository Structure
masa-framework/
├── core/
│   ├── state.py          # Shared state and circumplex definitions
│   ├── proxies.py        # Neuropsychological proxy estimation
│   └── memory.py         # Trajectory buffer and regulatory memory
├── agents/
│   └── agents.py         # Five-agent system with Identity Manifold
├── tests/
│   └── stress_tests.py   # All four stress test scenarios
├── graph.py              # LangGraph orchestration
├── main.py               # Demo runner
└── README.md
________________________________________
Collaboration
This project requires GPU access to reach Mode B. If you are working on:
•	Mechanistic interpretability with SAE tooling
•	Activation engineering or representation steering
•	Alignment evaluation frameworks
The orchestration layer is ready to accept real activation measurements. The architectural work is done. What we need is the instrumentation.
Contact via GitHub issues or the LinkedIn article linked below.
________________________________________
References
Anthropic Interpretability Team (2026). Emotion Concepts and their Function in a Large Language Model.
Russell, J.A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology.
Gross, J.J. (1998). Antecedent- and response-focused emotion regulation. JPSP.
Plutchik, R. (1980). A general psychoevolutionary theory of emotion.
Zhang et al. (2026). Dynamical Manifold Evolution Theory. arXiv:2505.20340.
Kantamneni, S. & Marks, S. (2026). Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations.
________________________________________
Built without institutional funding, GPU clusters, or academic affiliation. The architectural intuitions came from decades of study in neuroscience, psychology, and philosophy — not from the ML research tradition. That may be why it looks different.

