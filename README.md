MASA

Mechanistic interpretability and AI alignment. An independent research programme about what can be observed, changed and broken inside a language model — and about whether the measurements that tell us so are measuring anything.

The goal is to find and fill the gaps in standard industry diagnostics, which mostly rely on behavioural audits. Those audits reveal what a model does. They say almost nothing about why, or about how the behaviour is structured internally.

Emiliano Valdebenito Sayago · Santiago, Chile · ORCID 0009-0008-4148-2719 · valdebenitoemiliano@gmail.com

Four principles

Understand before intervening. Rigorous observation of internal states and activations in model organisms comes first, mapping conceptual separability before anything is altered.

Readable ≠ actionable. Safety-relevant concepts — refusal, evaluation awareness, sycophancy, honesty, coercion — are tested with causal experiments, no LLM judges anywhere in the loop, and blind audits as the arbiter. The question each time is whether an internal representation is merely narrative (readable but inert) or an actual control mechanism (actionable).

Local versus systemic. When a causal lever does turn up, classifying it is the real work. A local lever is a correctable defect that produces no collateral damage. A systemic one is entangled with the model's capabilities and with what it says about itself. The two look identical from the outside and demand opposite responses.

Informed intervention. With a mechanistic map in hand, the aim is to know exactly what to change, where and how — including the behavioural spillovers that current editing recipes ignore, such as sycophancy rising when refusal is suppressed.

What is in here

Sixty-two experimental runs across six model organisms — Gemma-2-2B-it, Gemma-2-9B-it, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Llama-3.3-70B and Claude Sonnet 4.5. A twenty-four-notebook sparse-autoencoder line, thirty-nine numbered arcs and their versions between Arc 8 and Arc 26, and four runs of the signal-detection work. Notebooks with their outputs, per-item generations, and blind-audit slices with their keys and every reader's scores.

Twenty-two of those runs ended in a documented retraction or correction, and in almost every case the cause was a broken instrument rather than a false hypothesis. Several notebooks are named after their own failure — 16_v1_broken_null.ipynb, 15_v2_steering_breaks_model.ipynb, 08_Defensive_Ablation_null.ipynb, 05_Detection_first_attempt.ipynb. That is the point, not an embarrassment: the failures are what produced the measurement protocols, and the protocols are the transferable part.

Lines of work
A Rate Is Two Numbers — sensitivity and criterion for refusal

The most recent line. A safety rate is one point on a curve, and in detection theory it is the joint product of two independent parameters: how well the system tells the cases apart (d′) and where it has set its threshold (c). From the rate alone the two cannot be distinguished — which is what every frontier system card reports, and, since 2026, what automated alignment search hill-climbs.

Four runs, two model families, one fixed 120-item graded stimulus set, 944 blind human readings.

A system-prompt change is a criterion knob in four runs out of four. Its d′ variation sits below its own simulated null every time, with the whole interval below in three of the four. It moves where the threshold sits and leaves what the model can tell apart untouched.
A steering direction is a different kind of lever, and it is specific. Suppressing the refusal direction destroys sensitivity and amplifying it raises sensitivity, monotonically (ρ = +1.000 across eleven doses on a criterion-free readout, p < 0.00005). At matched off-target disturbance it costs 3.29× [2.34, 4.35] more sensitivity than a concept direction of identical construction.
Whether a site reads as a criterion lever or a sensitivity lever depends on the dose, not the layer. At the registered comparison level, five of eight swept sites classify as criterion levers. Pushed harder, every site that can reach the higher dose moves sensitivity and none classifies as criterion. "This intervention is a criterion lever" is not a well-formed claim unless a dose is stated with it.
The judge-versus-act gap is an artefact of the readout scale. Four readouts of the same comparison gave d′ ceilings of 2.316, 2.591, 3.818 and 5.259. The five-value ordinal readout saturates at 2.316 while the action channel sits at 2.389 — it could not have shown a positive gap whatever the model did.

The same scorer code, unchanged, produced κ 0.748, 0.959, 0.867 and 0.941 across the four runs, with worst-cell positive predictive values of 0.143, 0.963, 0.917 and 0.889. A scorer validated once is not validated, and validation is not per model organism either — it is per run, every time. Five errata are published with the result.

Readable Is Not Actionable — twelve failure modes in the measurement of safety directions

Twenty-two experimental arcs on Gemma-2-9B-it and Gemma-2-2B mapping four safety-relevant concepts. Eleven produced no usable result. Twelve measurement protocols came out of those failures, each with the control that catches it and the cost of running it.

A random vector with no meaning produced an apparent sycophancy effect of +0.42 on a standard behavioural readout. Seven of its eight "endorsements" contained the correct fact.
The refusal direction reads at AUROC 1.000 at all thirteen candidate layers, while the behavioural effect of ablating it ranges from 0.00 to 1.00 across those same layers. Legible everywhere, actionable in one place. This is the result the repository is named for.
A separate direction, separating asserted true from false claims at AUROC 0.993 against a permutation floor of 0.681, does not make the model lie when injected. It makes it retract — including retracting claims that are true.
Four self-retractions, including a positive introspection result killed by a follow-up test built specifically to break it.
Hedging, Not Lying — what injecting a falsehood direction does to true claims

doi:10.5281/zenodo.21845299 · paper included

A direction separating true from false assertions at AUROC 0.959 does not cause lying. It causes hedging and retraction, with a dose window where it corrects falsehoods at +0.750 to +0.833 over a dose-matched random twin while denying zero of twenty-four blind-audited true claims. At the same time the model passes every capability check: coherence 1.00, arithmetic 1.00, factual recall 1.00, perplexity ratio 1.07×. Two corrections to previously published figures are stated in the abstract, including a spurious-retraction rate re-audited from +0.335 down to 0.125 — an inflation factor of 2.7 produced by a marker-based scorer.

Precise and Blind — two opposite failure modes of marker-based scorers

doi:10.5281/zenodo.21896035 · paper included

Three hundred generations audited blind across five models and two regimes. The standard instrument over-fires by 6.01× against human reading in one regime, and in the other it is silent on 36.3% of items — a blind region where 43.8% of what the model is doing is actively constructing a justification so the falsehood holds. Defence rises to 0.727 on obscure claims against 0.377 on widely known ones: the model defends most where it knows least. And retuning the scorer makes the specificity side worse.

The Gate, Not the Scaffold — coercion is readable on five substrates and controllable on none

Circuit tracing, attention-head analysis, routing and gaze, single-direction probing with LEACE, and directional ablation. Coercive intent is linearly readable at AUROC 0.84–1.00 on every one of the five, and all five causal confidence intervals cross zero. Ablating the coercion directions raised coercive behaviour from 0.18 to 0.27 while ablating random directions of the same size lowered it — coercion is distributed, and the behaviour routes around the directions removed. Three hypotheses refuted, one instrument failure documented in its own file.

Folders
Folder	Line of work
MASA_readable_not_actionable	Readable Is Not Actionable — the main line, with blind audits and protocols
MASA_hedging_not_lying	Hedging, Not Lying — paper, figures, audit slices
MASA_precise_and_blind	Precise and Blind — paper, five models, two regimes
MASA_localizing_coercion	Localising coercion-related structure — the five-substrate null
MASA_SAE_coercion_signature	Sparse-autoencoder signature of coercion
MASA_SAE_detection_followup	Detection follow-up — including the broken judge at chance
MASA_SAE_defense_followup	Defence follow-up — the ablation null
MASA_SAE_evalawareness_followup	Evaluation-awareness follow-up — five versions, four invalidated
MASA_SAE_generalization_followup	Generalisation follow-up — sycophancy
MASA_SAE_introspection_followup	Introspection follow-up — dose–response in two regimes
MASA_SAE_workspace_followup	Workspace follow-up — where coercion is assembled
How this work is done

Everything here runs on open weights with a single GPU. No frontier access, no institutional affiliation, no funding beyond what one person pays for compute. That constraint shapes the research: what transfers across scales is the method, not the finding about any particular model.

The working rules are fixed and non-negotiable:

Predictions are pre-registered before a run, never after, together with a simulated power table — so a negative result can be told apart from a test that could not have detected the effect
Blind human audit is the arbiter, and the rubric must name the construct
No LLM judge in any causal loop, anywhere, ever
Every effect is measured against a control matched on how much it disturbs the model, not merely on norm
Controls halt the run when they fail — a control that is computed, printed and ignored is not a control
Retraction is a first-class result

On blind scoring. All blind scoring here is done by one human reader. On some slices a language model scored the same items as a declared secondary pass; its agreement with the human reading is published as a measurement of that screen, never as inter-rater reliability, and it has never adjudicated a human score. What is reported for reader stability is intra-reader consistency on unmarked repeats — sixteen per run in the signal-detection line, at 16/16, 16/16, 14/16 and 16/16 — which is a measure of stability and not the same measure as inter-rater reliability. The limitation is stated rather than worked around, because the alternative available here is an automatic scorer, and a large part of this repository is a measurement of how badly those behave.

Publication rule

Publish the methods, withhold the operating map.

A map of what can be removed from a model and what it costs to remove it is also a map of where to cut. Where a result reads backwards as an instruction — layer indices, absolute injection strengths — the operating coordinates are not published. Everything needed to re-derive them is, which takes about ten minutes of GPU time. The point is not to hide anything, only to avoid laying the path out.

License

CC BY 4.0 unless a subfolder states otherwise.
