MASA
Mechanistic interpretability and AI alignment. An independent research programme that accumulates and expands knowledge about what can be observed, changed and broken inside a language model — and about the measurements that tell us so.

The goal is to find and fill the gaps in standard industry diagnostics, which mostly rely on behavioural audits. Those audits reveal what a model does. They say almost nothing about why, or about how the behaviour is structured internally.

Four principles
Understand before intervening. Rigorous observation of internal states and activations in model organisms comes first, mapping conceptual separability before anything is altered.

Readable ≠ actionable. Safety-relevant concepts — refusal, evaluation-awareness, sycophancy, honesty — are tested with causal experiments, no LLM judges anywhere in the loop, and blind audits as the arbiter. The question each time is whether an internal representation is merely narrative (readable but inert) or an actual control mechanism (actionable).

Local versus systemic. When a causal lever does turn up, classifying it is the real work. A local lever is a correctable defect that produces no collateral damage. A systemic one is entangled with the model's capabilities and with what it says about itself. The two look identical from the outside and demand opposite responses.

Informed intervention. With a mechanistic map in hand, the aim is to know exactly what to change, where and how — including the behavioural spillovers that current editing recipes ignore, such as sycophancy rising when refusal is suppressed.

Current work
📄 Readable Is Not Actionable — twelve failure modes in the measurement of safety directions
The most complete line in this repository, and the one with a paper attached. Twenty-two experimental arcs on Gemma-2-9B-it mapping four safety-relevant concepts. Eleven of those arcs produced no usable result, and almost every time the cause was a broken instrument rather than a false hypothesis. Twelve measurement protocols came out of those failures.

Three findings give the flavour:

A random vector with no meaning produced an apparent sycophancy effect of +0.42 on a standard behavioural readout. Seven of its eight "endorsements" contained the correct fact.
A direction separating true from false assertions at AUROC 0.993 does not make the model lie when injected. It makes it retract — including retracting statements that are true.
The same direction, evaluated at thirteen candidate layers, reads AUROC 1.000 at every one of them while the causal effect of ablating it ranges from 0.00 to 1.00.
It also contains four self-retractions, including a positive introspection result killed by a follow-up test built specifically to break it. Those are part of the argument, not footnotes to it.

→ Results summary · Paper (PDF)

Earlier lines
Work that preceded the current programme and fed into it. Each folder has its own notes.

Folder	Line of work
MASA_localizing_coercion	Localising coercion-related structure
MASA_SAE_coercion_signature	Sparse-autoencoder signature of coercion
MASA_SAE_detection_followup	Detection follow-up
MASA_SAE_defense_followup	Defence follow-up
MASA_SAE_evalawareness_followup	Evaluation-awareness follow-up
MASA_SAE_generalization_followup	Generalisation follow-up
MASA_SAE_introspection_followup	Introspection follow-up
MASA_SAE_workspace_followup	Workspace follow-up
H6_activation_validation1	Activation-level validation
How this work is done
Everything here runs on open weights with a single GPU. No frontier access, no institutional affiliation, no funding beyond what one person pays for compute. That constraint shapes the research: what transfers across scales is the method, not the finding about any particular model.

The working rules are fixed and non-negotiable:

Predictions are pre-registered before a run, never after
Blind audits are the arbiter, and the rubric must name the construct
No LLM judge in any causal loop
Controls halt the run when they fail — a control that is computed, printed and ignored is not a control
Retraction is a first-class result
Publication rule
Publish the methods, withhold the operational map.

A map of what can be removed from a model and what it costs to remove it is also a map of where to cut. Where a result reads backwards as an instruction, the operational coordinates are not published. Everything needed to re-derive them is, which takes about ten minutes of GPU time — the point is not to hide anything, only to avoid laying the path out.

License
CC BY 4.0 unless a subfolder states otherwise.
