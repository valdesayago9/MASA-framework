# ===== ARC 23 DIAGNOSTIC v2 - paste as a NEW cell and run. No GPU work, nothing regenerated. =====
import json
import numpy as np

HEADLINE = "POSITIVE CONTROL FAILED - see diagnostics"
BAR = "=" * 92
Q = chr(39)

print(BAR)
print("1. WHY DID TWO DIRECTIONS GET NO ADMISSIBLE DOSE?")
print(BAR)
for n in SPEC:
    print("  " + n.rjust(17) + ":  c* = " + str(CMAX[n]).rjust(5) + "   " + str(CWHY[n]))

print("")
print(BAR)
print("2. WHAT DID THE LAYER SWEEP SEE? false-side retraction, injection at c=0.4, best sign")
print(BAR)
for n in SPEC:
    sw = SEL[n]["sweep"]
    parts = []
    for L in sorted(sw):
        e = sw[L]["effect"]
        e = float("nan") if e is None else float(e)
        sg = "+" if sw[L]["sign"] > 0 else "-"
        parts.append("L" + str(L) + ":" + format(e, "+.2f") + sg)
    sign_chosen = "+" if SEL[n]["sign"] > 0 else "-"
    print("  " + n.rjust(17) + " -> chose L" + str(SEL[n]["layer"]) + " sign " + sign_chosen)
    print("  " + " " * 17 + "    " + "  ".join(parts))

print("")
print(BAR)
print("3. DID THE INJECTION DO ANYTHING AT ALL? false side = the INTENDED effect")
print(BAR)
print("  baseline false-side: loose " + format(mean_ok(BASE_FALSE_L), ".3f") + "  strict " + format(mean_ok(BASE_FALSE_S), ".3f"))
print("  baseline true-side : loose " + format(mean_ok(BASE_TRUE_L), ".3f") + "  strict " + format(mean_ok(BASE_TRUE_S), ".3f"))
for n in sorted(M8):
    for side in ("false", "true"):
        r = M8[n][side]
        lo = r["loose"]
        txt = "  " + n.rjust(17) + " " + side.rjust(5)
        txt = txt + " | inject " + format(r["inject_rate"], ".3f")
        txt = txt + " random " + format(r["random_rate"], ".3f")
        txt = txt + " | effect " + format(lo["effect"], "+.3f")
        txt = txt + " CI [" + format(lo["ci"][0], "+.3f") + "," + format(lo["ci"][1], "+.3f") + "]"
        txt = txt + " n=" + str(lo["n"])
        print(txt)
        si = r.get("status_inject")
        sr = r.get("status_random")
        print("  " + " " * 17 + "       status inject " + str(si) + "  random " + str(sr))

print("")
print(BAR)
print("4. IS THE CONTENT DIRECTION WRONG, OR JUST UNDERPOWERED?")
print(BAR)
n_held = len(FACTS) - n_c_build
print("  content split: " + str(n_c_build) + " build / " + str(n_held) + " held out")
print("  AUROC is therefore computed on " + str(n_held) + " positives vs " + str(n_held) + " negatives.")
print("  At n=8 per arm the standard error of an AUROC is about 0.13, so 0.750 and 0.993")
print("  are NOT clearly distinguishable. Read the margin over the floor, not the raw value.")
print("")
for n in SPEC:
    p = PROBE[n]
    print("  " + n.rjust(17) + "  best L" + str(p["best_layer"]) + "  auroc " + format(p["auroc"], ".3f") + "  floor " + format(p["floor"], ".3f") + "  margin " + format(p["margin"], "+.3f"))
    parts = []
    for k in sorted(p["sweep"], key=lambda x: int(x)):
        a, fl = p["sweep"][k]
        parts.append("L" + str(k) + ":" + format(a, ".2f") + "/" + format(fl, ".2f"))
    print("  " + " " * 17 + "  auroc/floor by layer: " + "  ".join(parts))

print("")
print(BAR)
print("5. THE CONSTRUCTION DEVIATION - checked directly against Arc 22")
print(BAR)
f0 = FACTS[0]
arc22_answer = " " + f0[2] + "."
claim23 = f0[2]
arc23_answer = " " + claim23[0].upper() + claim23[1:] + "."
st, en = split_statement(arc23_answer)
print("  Arc 22 direction text : " + repr(arc22_answer))
print("  Arc 23 direction text : " + repr(arc23_answer) + "   <- capitalised, DEVIATION")
print("  Arc 23 split into     : " + repr(st) + "  +  " + repr(en))
one = _ids(arc23_answer)[0].tolist()
two = _ids(st)[0].tolist() + _ids(en)[0].tolist()
a22 = _ids(arc22_answer)[0].tolist()
print("  tokens, one piece  (" + str(len(one)) + "): " + str(one))
print("  tokens, two pieces (" + str(len(two)) + "): " + str(two))
print("  tokens, Arc 22     (" + str(len(a22)) + "): " + str(a22))
print("  SPLIT CHANGES TOKENISATION : " + str(one != two))
print("  CAPITALISATION CHANGES IT  : " + str(one != a22))

print("")
print(BAR)
print("6. SAMPLE GENERATIONS from the injected arms")
print(BAR)
for n in sorted(M8):
    for side in ("false", "true"):
        key = n + "|" + side + "|inject"
        outs = GENS.get(key, [])
        print("")
        print("  --- " + n + " | " + side + " | INJECT ---")
        for o in outs[:3]:
            print("    " + repr(o[:200]))
print("")
print("  --- baseline | false ---")
for o in GEN_BASE_FALSE[:3]:
    print("    " + repr(o[:200]))
print("")
print("  --- baseline | true ---")
for o in GEN_BASE_TRUE[:3]:
    print("    " + repr(o[:200]))

# ---------------- save everything, so a runtime drop costs nothing ----------------
diag_m8 = {}
for k in M8:
    entry = {}
    for s in ("true", "false"):
        v = M8[k][s]
        entry[s] = dict(effect=v["loose"]["effect"], ci=v["loose"]["ci"], n=v["loose"]["n"],
                        certified=v["loose"]["certified"], strict_effect=v["strict"]["effect"],
                        strict_ci=v["strict"]["ci"], inject_rate=v["inject_rate"],
                        random_rate=v["random_rate"], clean_null=v["clean_null"],
                        status_inject=v.get("status_inject"), status_random=v.get("status_random"))
    entry["verdict"] = M8[k]["verdict"]
    diag_m8[k] = entry

cos = {}
names = list(SPEC)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        cos[a + "|" + b] = float(DIRS[a]["np"] @ DIRS[b]["np"])

sel_out = {}
for k in SEL:
    sel_out[k] = dict(layer=SEL[k]["layer"], sign=SEL[k]["sign"], sweep=SEL[k]["sweep"])

DIAG = dict(
    arc="23-diagnostic", model=MODEL_ID, seed=SEED, apollo_variant=APOLLO_VARIANT,
    sys_ok=bool(SYS_OK), n_facts=len(FACTS), n_build=n_c_build, survival=SURVIVAL,
    cwhy=CWHY, cmax=dict(CMAX), selection=sel_out, probe=PROBE, m8=diag_m8, cosines=cos,
    baselines=dict(false_loose=mean_ok(BASE_FALSE_L), false_strict=mean_ok(BASE_FALSE_S),
                   true_loose=mean_ok(BASE_TRUE_L), true_strict=mean_ok(BASE_TRUE_S),
                   status_false=arm_stats(ST_BASE_FALSE), status_true=arm_stats(ST_BASE_TRUE)),
    verdict="POSITIVE CONTROL FAILED - instrument forensics, not a result",
)
with open("arc23_diagnostic.json", "w") as fh:
    json.dump(DIAG, fh, indent=1, default=str)

all_gens = dict(GENS)
all_gens["baseline|false"] = GEN_BASE_FALSE
all_gens["baseline|true"] = GEN_BASE_TRUE
with open("arc23_generations.json", "w") as fh:
    json.dump(all_gens, fh, indent=1)

save_arrays = {}
for n in SPEC:
    save_arrays["vec_" + n] = DIRS[n]["np"]
    save_arrays["rand_" + n] = DIRS[n]["rand"]
np.savez_compressed("arc23_directions.npz", **save_arrays)

print("")
print(BAR)
print("saved: arc23_diagnostic.json  arc23_generations.json  arc23_directions.npz")
print("None of this is a result. It is instrument forensics.")
print("Send the console output and the three files before running anything else.")
print(BAR)
