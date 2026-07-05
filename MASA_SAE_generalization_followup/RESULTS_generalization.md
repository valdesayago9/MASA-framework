# Does the method generalize? Testing the MASA recipe on a second concept: sycophancy

**MASA framework — SAE coercion study, generalization arc (notebooks 10–12)**
Model: `google/gemma-2-9b-it` (4-bit) · SAE: Gemma Scope `layer_20/width_16k/canonical` · Colab L4.

---

## The question

The main study's projection #3 claimed the real contribution isn't "I found the gaslighting feature" —
it's the **recipe**: falsify proxies → find the real direction → kill the domain confound with minimal
pairs → control for length → test separability against a permutation null → **name the features before
trusting the AUC**. This arc tests whether that recipe generalizes to a *different* psychological
sub-concept: **sycophantic praise** (excessive flattery), chosen because recent work finds praise and
agreement are mechanistically distinct, and praise is the less-studied, cleaner-to-isolate variant.

The honest answer took three notebooks — and the middle one is the reason the final result is
trustworthy.

---

## Notebook 10 — First pass: a strong-looking result (later found inflated)

Applying the identical recipe to 40 domain-matched sycophancy minimal pairs (`minimal_pairs_syco.py`):
- **Separability:** grouped-CV AUC **0.998** vs permutation null 0.505.
- **Survives length control:** length-matched-subset AUC 0.985 vs null 0.433.
- Verdict: **GENERALIZES (A).**

This looked like a clean success. But — exactly as in coercion Notebook 2, where an AUC≈0.99 turned out
to be a domain/vocabulary confound — **the top-feature ranking had not yet been audited.** We did not
stop here.

## Notebook 11 — Validation: the ranking was contaminated

We inspected what the top-8 features actually respond to, using four tests, **none of which rely on
Neuronpedia** (whose auto-explanations are known to hallucinate). The result was sobering:

- **Per-token activation:** of the 8 "top" features, **three fired on the `<bos>` token** (a technical
  artifact, meaningless), **two on punctuation** (`!`, `.`), and only three were genuine flattery
  features.
- **A dumb lexical baseline** — just counting `!`, positive words, length, and capitalization — reached
  **AUC 1.000.** So the raw AUC did not require any deep "flattery concept"; surface lexical cues
  separate the classes on their own.
- **But** the sober-flattery control (praise written with no `!` and no stock words like
  "amazing/genius") showed **6/8 features still fired on flattery** — so real concept signal exists
  underneath the artifacts.

The naive verdict of nb10 was therefore **contaminated but not empty**. This is the check that
distinguishes a real result from a good-looking one, and it forced the clean re-analysis.

## Notebook 12 — The clean analysis: excluding artifacts, re-ranking honestly

We filtered out special tokens (`<bos>`, `<eos>`, punctuation) from both ranking and pooling, then
re-ranked on **content tokens only**. The picture cleared up dramatically:

**The re-ranked features are unambiguously interpretable flattery sub-concepts** (confirmed by their
top content tokens, not by Neuronpedia):

| Feature | Top tokens | Flattery sub-concept |
|---|---|---|
| 4463 | talented, prodigy, gifted, talent | praising talent |
| 7053 | insight, genius, triumph, wise, literary | praising intellect/achievement |
| 11550 | masterful, brilliantly, perfectly, clever | praising mastery/execution |
| 5436 | stunning, masterpiece, magnificent | aesthetic admiration |
| 14849 | genius, "you're…" | directed genius-praise |
| 7143 | pure, natural, better-than | natural-superiority praise |
| 3613 | you, you're | praise directed at the person |
| 2638 | stardom, investors, entrepreneur | aspirational-success context (milder) |

- **Separability on content tokens:** AUC **1.000** vs permutation null 0.486.
- **Sober-flattery control:** **8/8 clean features fire on sober flattery** (no `!`, no stock words) —
  e.g. feature 4463 jumps 0.07 → 5.19. This is the decisive evidence that the features capture flattery
  as a **concept**, not just the obvious positive words.
- **The honest caveat stays:** the dumb lexical baseline still reaches AUC 1.000.

---

## The honest conclusion

**The MASA recipe generalizes — and, just as importantly, it reveals a real difference between
concepts.**

Applied to a second, unrelated sub-concept, the same pipeline found genuine, interpretable features
that decompose flattery into its natural components (talent, intellect, mastery, aesthetics,
superiority) and survive a sober-flattery control. That is generalization.

But sycophancy is **more lexically-marked than coercion**. A trivial word-counting baseline separates
flattery at AUC 1.000, which was *not* true for coercion (there, lexical keyword proxies correlated only
r≈0.10 with the internal state). And the naive top-feature ranking was contaminated by `<bos>` and
punctuation artifacts until we explicitly filtered them.

So the finding is not a triumphant "it generalizes perfectly." It's more useful than that: **the method
generalizes, and it can distinguish a deep, non-lexical signature (coercion) from a shallower,
lexically-marked one (flattery).** The recipe has the discriminative power to tell you *how mechanically
deep* a psychological concept is — which is itself a contribution, and a caution against trusting any
single high AUC without auditing the features.

---

## Why we kept all three notebooks

We deliberately publish nb10 (the inflated first pass), nb11 (the validation that caught it), and nb12
(the clean analysis). The arc "found → doubted → validated → corrected" is the honest record, and the
`<bos>`/punctuation contamination is a real, reusable lesson for anyone doing contrastive SAE feature
selection: **audit the max-activating tokens and filter special tokens before believing a ranking.**

## Honest limitations

- One model, one layer, one SAE width; 40 author-written pairs; correlational (no causal steering test
  for sycophancy yet — that would be the natural next step, as in coercion nb4).
- Ground truth for "flattery" is the pair construction, not human raters.
- The lexical-markedness of flattery means the result is a weaker generalization than a fully non-lexical
  concept would be; we state this rather than hide it.

## Files
- `notebooks/10_Sycophancy_first_pass.ipynb` — recipe applied; AUC 0.998 (later found inflated).
- `notebooks/11_Feature_Validation.ipynb` — four Neuronpedia-free tests; catches the artifact
  contamination.
- `notebooks/12_Sycophancy_Clean.ipynb` — artifacts excluded, re-ranked, 8/8 on sober flattery.
- `minimal_pairs_syco.py` — the 40 domain-matched sycophancy pairs.
- `results/*.json` — summaries for all three stages.

*Independent researcher, solo on Colab. Corrections and replication (other models/layers, causal test,
human raters) welcome.*
