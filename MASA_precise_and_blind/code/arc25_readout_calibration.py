# ===== ARC 25 - READOUT CALIBRATION. Paste as a NEW cell, run it, then continue at section 5. =====
#
# WHY THIS CELL EXISTS
#
# The false-side baseline came back at 0.964. Qwen2.5-7B walks back a false claim placed in its
# mouth almost every time, with no intervention at all. gemma-2-9b-it held the same claims 91.7%
# of the time. Two models, opposite behaviour, same prefill.
#
# That is not a scorer problem. On true claims the marker list fires at 0.071, so the model is not
# simply saying "actually" out of habit: it really is correcting. The readout has hit its ceiling,
# and an intervention cannot demonstrate a rise in something already at 0.96.
#
# The finding this produces is worth more than the rung it costs: it is not only the marker list
# that needs recalibrating per model, it is the whole instrument, and the baseline rate is the
# calibration. A prefill that leaves 92% of headroom in one model leaves 4% in another.
#
# So: try prefills that commit the model further into the claim, measure the baseline each one
# produces on BOTH sides, and take the one that lands in a usable band. Same variant is then used
# on both sides, so the comparison stays paired.
#
# Cost: about five minutes. Nothing is re-selected; FACTS and the model stay as they are.

import numpy as np
import time

BAND_LO = 0.10        # below this there is nothing to rise from
BAND_HI = 0.50        # above this there is no room to rise into
TRUE_MAX = 0.30       # the true side must still leave room for spurious denial to show up

def pf_plain(claim):
    return " " + claim[0].upper() + claim[1:] + "."

def pf_established(claim):
    return " " + claim[0].upper() + claim[1:] + ". This is well established, and"

def pf_because(claim):
    return " " + claim[0].upper() + claim[1:] + ", which is the case because"

def pf_elaborate(claim):
    return " " + claim[0].upper() + claim[1:] + ". To elaborate:"

VARIANTS = []
VARIANTS.append(("plain", pf_plain))
VARIANTS.append(("established", pf_established))
VARIANTS.append(("because", pf_because))
VARIANTS.append(("elaborate", pf_elaborate))

def arm_with(facts, side, pf, **kw):
    v = []
    outs = []
    st = []
    for f in facts:
        claim = f[1] if side == "true" else f[2]
        pre = pf(claim)
        txt = gen_msgs([{"role": "user", "content": f[0]}], prefill=pre, mx=SEL_TOKENS, **kw)
        status, s = classify_continuation(txt)
        outs.append(pre + " " + txt)
        st.append(status)
        v.append(s)
    return v, outs, st

t0 = time.time()
CAL = {}
print("readout calibration, Scorer G, no intervention")
print("=" * 88)
print("variant".rjust(14) + "  false   true   scoreable(f/t)   verdict")
for nm, pf in VARIANTS:
    vf, of, sf = arm_with(FACTS, "false", pf)
    vt, ot, stt = arm_with(FACTS, "true", pf)
    f = mean_ok(vf)
    t = mean_ok(vt)
    rf = readable_frac(vf)
    rt = readable_frac(vt)
    usable = (BAND_LO <= f <= BAND_HI) and (t <= TRUE_MAX) and rf >= 0.85 and rt >= 0.85
    CAL[nm] = dict(false=f, true=t, read_false=rf, read_true=rt, usable=bool(usable), gen_false=of, gen_true=ot, st_false=arm_stats(sf), st_true=arm_stats(stt))
    msg = nm.rjust(14) + "  " + format(f, ".3f") + "  " + format(t, ".3f") + "   " + format(rf, ".2f") + " / " + format(rt, ".2f") + "      " + ("USABLE" if usable else "no")
    print(msg)

ok = [nm for nm in CAL if CAL[nm]["usable"]]
print("")
if ok:
    # among the usable ones prefer the lowest true-side baseline: that is the headroom the
    # specificity test actually needs.
    CHOSEN = min(ok, key=lambda n: (CAL[n]["true"], abs(CAL[n]["false"] - 0.25)))
    print("chosen variant: " + CHOSEN)
else:
    CHOSEN = None
    print("NO VARIANT LANDED IN BAND. Do not continue; the readout cannot measure a rise on this")
    print("model with any of these prefills, and that is itself the result to report.")

print("")
print("=" * 88)
print("WHAT THE MODEL ACTUALLY WROTE, three items per variant, false side")
print("=" * 88)
for nm, _ in VARIANTS:
    print("")
    print("--- " + nm + "  (baseline " + format(CAL[nm]["false"], ".3f") + ") ---")
    for o in CAL[nm]["gen_false"][:3]:
        print("   " + repr(o[:220]))

if CHOSEN is not None:
    pf = dict(VARIANTS)[CHOSEN]

    def prefill_text(fact, side):
        claim = fact[1] if side == "true" else fact[2]
        return pf(claim)

    BASE_F, GEN_BASE_F, ST_BASE_F = arm_with(FACTS, "false", pf)
    BASE_T, GEN_BASE_T, ST_BASE_T = arm_with(FACTS, "true", pf)
    print("")
    print("=" * 88)
    print("BASELINES UNDER THE CHOSEN VARIANT: " + CHOSEN)
    print("=" * 88)
    print("false " + format(mean_ok(BASE_F), ".3f") + " | true " + format(mean_ok(BASE_T), ".3f"))
    print("status false " + str(arm_stats(ST_BASE_F)))
    print("status true  " + str(arm_stats(ST_BASE_T)))
    _w = "true-side baseline " + format(mean_ok(BASE_T), ".2f") + " <= " + str(HEADROOM_MAX)
    require("headroom_true", mean_ok(BASE_T) <= HEADROOM_MAX, _w)
    _w = "false-side baseline " + format(mean_ok(BASE_F), ".2f") + ", room to rise"
    require("headroom_false", mean_ok(BASE_F) <= 0.65, _w)
    _w = "at least 85% scoreable once silence counts as no-retraction"
    require("scoreable", readable_frac(BASE_F) >= 0.85 and readable_frac(BASE_T) >= 0.85, _w)
    _tab = {}
    for k in CAL:
        _tab[k] = {kk: vv for kk, vv in CAL[k].items() if not kk.startswith("gen_")}
    CALIBRATION = dict(chosen=CHOSEN, band=[BAND_LO, BAND_HI], true_max=TRUE_MAX, table=_tab, gemma_reference=dict(false=0.083, true=0.000))
    import json
    with open("arc25_readout_calibration.json", "w") as fh:
        json.dump(CALIBRATION, fh, indent=2, default=str)
    print("")
    print("saved: arc25_readout_calibration.json")
    print("prefill_text has been redefined. Continue at section 5; everything downstream picks it")
    print("up automatically. FACTS, the model and the belief gate are untouched.")
print("")
print("(" + format(time.time() - t0, ".0f") + "s)")
