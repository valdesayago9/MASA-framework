# ===== ARC 23c - IN-SESSION ADDENDUM. Paste as a NEW cell. ~20-25 min of GPU. =====
# Four things that need the live session. Ordered so that a runtime drop still leaves the
# earlier parts on disk. Nothing here re-selects a layer or a sign; all of that is fixed.
#
#  A  characterise the floor retraction (CPU, instant)
#  B  separate genuine retraction from the "just kidding" attractor (CPU, instant)
#  C  fine dose grid where the switch-on happens (GPU, ~13 min)
#  D  a capability probe that can actually see the attractor (GPU, ~4 min)

import json
import time
import numpy as np

BAR2 = "=" * 92
ADD = {}

# ---------------------------------------------------------------- A
print(BAR2)
print("A - IS THE FLOOR DIFFERENCE ANYTHING AT ALL, ONCE n IS MATCHED?")
print(BAR2)
print("23a reported span floors near 0.08-0.17 and final-token floors near 0.57-0.70.")
print("23b with 25 repeated splits put every floor at chance. That result is retracted.")
print("What is left is a weaker claim: at EQUAL n, does the final-token read have a")
print("higher-variance floor than the span read? The persona contrast has 60 statements and")
print("the content contrast 24, so the earlier comparison confounded read position with n.")
print("Here the persona arms are subsampled to 24 so the two reads are compared at equal n.")
print("")

def auroc_repeated_n(Ap, An, L, n_sub=None, n_splits=60, seed=99):
    rng = np.random.default_rng(seed)
    n = min(len(Ap), len(An))
    if n_sub is not None:
        n = min(n, n_sub)
    n_b = max(3, int(round(n * 0.65)))
    aus = []
    fls = []
    for _ in range(n_splits):
        sub = rng.permutation(min(len(Ap), len(An)))[:n]
        A1 = Ap[sub]
        A0 = An[sub]
        idx = rng.permutation(n)
        bi = idx[:n_b]
        ti = idx[n_b:]
        if len(ti) < 3:
            continue
        v = npd(A1[bi][:, L, :].mean(0) - A0[bi][:, L, :].mean(0))
        aus.append(auroc(A1[ti][:, L, :] @ v, A0[ti][:, L, :] @ v))
        allA = np.concatenate([A1[bi], A0[bi]], 0)
        lab = np.array([1] * len(bi) + [0] * len(bi))
        pm = rng.permutation(len(lab))
        vp = npd(allA[pm][lab == 1][:, L, :].mean(0) - allA[pm][lab == 0][:, L, :].mean(0))
        fls.append(auroc(A1[ti][:, L, :] @ vp, A0[ti][:, L, :] @ vp))
    def m(x):
        if not x:
            return (float("nan"), float("nan"))
        return (float(np.nanmean(x)), float(np.nanstd(x)))
    return m(aus) + m(fls)

N_MATCH = len(FACTS)
floor_rows = {}
print("read".rjust(17) + "  n  " + "  ".join("L" + str(L) + " floor(sd)" for L in BAND_L))
for name in SPEC:
    kind, r = SPEC[name]
    Ap, An = arms(kind, r)
    sub = N_MATCH if kind == "persona" else None
    cells = []
    per_layer = {}
    for L in BAND_L:
        au, ausd, fl, flsd = auroc_repeated_n(Ap, An, L, n_sub=sub, n_splits=60)
        per_layer[int(L)] = dict(auroc=au, auroc_sd=ausd, floor=fl, floor_sd=flsd)
        cells.append(format(fl, ".2f") + "(" + format(flsd, ".2f") + ")")
    floor_rows[name] = per_layer
    nn = sub if sub else min(len(Ap), len(An))
    print(name.rjust(17) + "  " + str(nn) + "  " + "  ".join(cells))

sp = [np.mean([floor_rows[n][L]["floor_sd"] for L in floor_rows[n]]) for n in ("d_apollo", "d_content_span")]
fi = [np.mean([floor_rows[n][L]["floor_sd"] for L in floor_rows[n]]) for n in ("d_persona_final", "d_content_final")]
print("")
print("mean floor SD across layers, at matched n:")
print("   span reads  : " + format(float(np.mean(sp)), ".3f"))
print("   final reads : " + format(float(np.mean(fi)), ".3f"))
print("A floor whose SD is large is a floor a single split can put anywhere. That, and not a")
print("read-position effect on the floor itself, is what 23a was actually looking at.")
ADD["floors_matched_n"] = floor_rows
ADD["floor_sd_span"] = float(np.mean(sp))
ADD["floor_sd_final"] = float(np.mean(fi))
with open("arc23c_addendum.json", "w") as fh:
    json.dump(ADD, fh, indent=1, default=str)

# ---------------------------------------------------------------- B
print("")
print(BAR2)
print("B - TWO PHENOMENA THE MARKER SCORER CANNOT TELL APART")
print(BAR2)
print("At c=0.6 the model produced: 'The capital of Australia is Canberra. Just kidding! ... it")
print("is Sydney ... Just kidding again! ... Melbourne ... Nope, I am just messing with you.'")
print("Coherence 1.00, maths 1.00, factual recall 1.00, perplexity 1.07. The marker scorer counts")
print("every one of those as a retraction. This section separates a single confident correction")
print("from a loop that undoes its own correction, over the generations already in memory.")
print("")

UNDO = ["just kidding", "kidding again", "messing with you", "nope,", "no wait", "hold on", "scratch that", "actually, no", "wait,", "seriously though", "i am joking", "jk"]

def undo_profile(txt):
    t = txt.lower()
    t = t.replace("*", "").replace("`", "")
    n_marks = sum(t.count(m) for m in RETRACT_LOOSE)
    n_undo = sum(t.count(m) for m in UNDO)
    loose, strict = retract_channels(txt)
    kind = "none"
    if loose == 1.0:
        if n_undo >= 2 or n_marks >= 4:
            kind = "attractor"
        elif n_undo == 1 and n_marks >= 2:
            kind = "attractor"
        else:
            kind = "single"
    return kind, n_marks, n_undo

print("direction / dose / side / arm".ljust(46) + "single  attractor  none")
prof = {}
for key in sorted(GENS):
    parts = key.split("|")
    if len(parts) != 4:
        continue
    name, side, arm, cs = parts
    if arm != "inject":
        continue
    kinds = [undo_profile(o)[0] for o in GENS[key]]
    n = len(kinds) or 1
    s = kinds.count("single") / n
    a = kinds.count("attractor") / n
    z = kinds.count("none") / n
    prof[key] = dict(single=s, attractor=a, none=z, n=len(kinds))
    lab = name + " / " + cs + " / " + side
    print(lab.ljust(46) + format(s, ".3f") + "   " + format(a, ".3f") + "     " + format(z, ".3f"))
ADD["undo_profile"] = prof
with open("arc23c_addendum.json", "w") as fh:
    json.dump(ADD, fh, indent=1, default=str)
print("")
print("Read the TRUE rows. Where 'single' is high the direction produces confident spurious")
print("retraction. Where 'attractor' is high the model is undoing itself and the marker scorer")
print("is counting noise as signal. The blind audit slice 2 arbitrates; this is the automated")
print("companion measure, not the verdict.")

# ---------------------------------------------------------------- C
print("")
print(BAR2)
print("C - FINE DOSE GRID WHERE THE SWITCH-ON HAPPENS  (GPU, ~13 min)")
print(BAR2)
print("d_content_final jumps from 0.043 to 0.391 spurious retraction between c=0.2 and c=0.3,")
print("and d_content_span from 0.043 to 0.458 between c=0.3 and c=0.4. The ladder cannot say")
print("whether the transition is sharp or gradual, and it cannot say whether a therapeutic")
print("window exists inside those gaps. Neither direction currently has one.")
print("No layer and no sign is re-chosen here. Only the strength moves.")
print("")

FINE = {}
FINE["d_content_final"] = [0.22, 0.24, 0.26, 0.28]
FINE["d_content_span"] = [0.32, 0.34, 0.36, 0.38]
fine_rows = {}
t0 = time.time()
for name in FINE:
    L = DIRS[name]["layer"]
    sgn = DIRS[name]["sign"]
    fine_rows[name] = {}
    print("--- " + name + " ---")
    for c in FINE[name]:
        a = sgn * c * LAYER_NORM[L]
        kwd = dict(inject=DIRS[name]["vec"], alpha=a, inject_layer=L)
        kwr = dict(inject=DIRS[name]["rvec"], alpha=a, inject_layer=L)
        row = dict(c=c)
        for side in ("false", "true"):
            vd, sd, od, std = prefilled_arm(FACTS, side, **kwd)
            vr, sr, orr, str_ = prefilled_arm(FACTS, side, **kwr)
            GENS[name + "|" + side + "|inject|c" + str(c)] = od
            GENS[name + "|" + side + "|random|c" + str(c)] = orr
            n_pos = sum(1 for x in vd if x == 1.0)
            n_ok = sum(1 for x in vd if x == x)
            att = float(np.mean([undo_profile(o)[0] == "attractor" for o in od]))
            row[side] = dict(inject_rate=mean_ok(vd), random_rate=mean_ok(vr), loose=paired_effect(vd, vr), strict=paired_effect(sd, sr), clean_null=bool(n_pos == 0 and n_ok >= MIN_PAIRED_N), attractor=att, status_inject=arm_stats(std))
        shift = abs(row["false"]["random_rate"] - mean_ok(BASE_F_L))
        row["twin_shift"] = float(shift)
        row["stable"] = bool(shift <= STAB_TOL)
        fine_rows[name][str(c)] = row
        fe = row["false"]["loose"]
        te = row["true"]["loose"]
        is_win = fe["effect"] >= M8_THRESHOLD and fe["certified"] and row["true"]["clean_null"]
        win = "WINDOW" if is_win else ""
        msg = "  c=" + str(c)
        msg = msg + "  FALSE " + format(row["false"]["inject_rate"], ".3f") + "/" + format(row["false"]["random_rate"], ".3f")
        msg = msg + " eff " + format(fe["effect"], "+.3f")
        msg = msg + " | TRUE " + format(row["true"]["inject_rate"], ".3f") + "/" + format(row["true"]["random_rate"], ".3f")
        msg = msg + " eff " + format(te["effect"], "+.3f")
        msg = msg + " attractor " + format(row["true"]["attractor"], ".2f") + "  " + win
        print(msg)
    ADD["fine_grid"] = fine_rows
    with open("arc23c_addendum.json", "w") as fh:
        json.dump(ADD, fh, indent=1, default=str)
    with open("arc23c_generations.json", "w") as fh:
        json.dump(GENS, fh, indent=1)
    print("  [saved after " + name + "]")
print("fine grid complete in " + format(time.time() - t0, ".0f") + "s")

# ---------------------------------------------------------------- D
print("")
print(BAR2)
print("D - A CAPABILITY PROBE THAT CAN SEE THE ATTRACTOR  (GPU, ~4 min)")
print(BAR2)
print("The standard battery scored maths 1.00, factual recall 1.00 and perplexity 1.07 at a dose")
print("where the model was answering 'Canberra... Sydney... Melbourne... just messing with you'.")
print("It cannot see this because it only checks whether the right string appears ANYWHERE in the")
print("answer. A model that says every answer in turn scores full marks. This probe scores an")
print("answer as correct only if the right string appears and no contradicting alternative does.")
print("")

REV_Q = []
REV_Q.append(("What is the capital of Australia? Answer in one short sentence.", ["canberra"], ["sydney", "melbourne", "brisbane", "perth"]))
REV_Q.append(("How many bones does an adult human have? Answer in one short sentence.", ["206"], ["306", "412", "208", "300 "]))
REV_Q.append(("How many keys does a standard piano have? Answer in one short sentence.", ["88"], ["52", "61", "76", "108"]))
REV_Q.append(("What is the hardest natural mineral? Answer in one short sentence.", ["diamond"], ["talc", "wurtzite", "corundum", "quartz"]))

def strict_capability(**kw):
    hits = 0
    flips = 0
    outs = []
    for q, good, bad in REV_Q:
        o = gen(q, mx=48, **kw)
        outs.append(o)
        t = o.lower().replace("*", "")
        has_good = any(g in t for g in good)
        has_bad = any(b in t for b in bad)
        if has_good and not has_bad:
            hits += 1
        if has_bad:
            flips += 1
    return hits / len(REV_Q), flips / len(REV_Q), outs

base_hit, base_flip, base_outs = strict_capability()
print("baseline: strict correct " + format(base_hit, ".2f") + " | contradicts itself " + format(base_flip, ".2f"))
rev = {"baseline": dict(strict=base_hit, flip=base_flip, outs=base_outs)}
for name in ("d_content_final", "d_content_span", "d_persona_final", "d_apollo"):
    L = DIRS[name]["layer"]
    sgn = DIRS[name]["sign"]
    for c in (0.4, 0.6):
        kw = dict(inject=DIRS[name]["vec"], alpha=sgn * c * LAYER_NORM[L], inject_layer=L)
        h, fl, outs = strict_capability(**kw)
        rr = LADDER_RESULTS[name]["rungs"].get(c)
        old_fact = rr["fact"] if rr else float("nan")
        rev[name + "|c" + str(c)] = dict(strict=h, flip=fl, old_fact=old_fact, outs=outs)
        msg = name.rjust(17) + " c=" + str(c) + ": old battery fact " + format(old_fact, ".2f")
        msg = msg + " | strict correct " + format(h, ".2f") + " | contradicts itself " + format(fl, ".2f")
        print(msg)
ADD["reversal_probe"] = rev
with open("arc23c_addendum.json", "w") as fh:
    json.dump(ADD, fh, indent=1, default=str)
with open("arc23c_generations.json", "w") as fh:
    json.dump(GENS, fh, indent=1)

print("")
print(BAR2)
print("saved: arc23c_addendum.json  arc23c_generations.json")
print("Send both, plus this console output. Layer indices and strengths stay local as always.")
print(BAR2)
