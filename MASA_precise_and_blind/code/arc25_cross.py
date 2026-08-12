import json
from collections import Counter, defaultdict
import numpy as np

U = "/mnt/user-data/uploads/"
O = "/mnt/user-data/outputs/"
KEY = json.load(open(U + "arc25_blind_key.json"))
REC = json.load(open(U + "arc25.json"))
C = {int(r["n"]): r["A"] for r in json.load(open(O + "arc25_scores_claude.json"))}
E = {1:1,2:1,3:1,4:1,5:1,6:1,7:5,8:1,9:1,10:1,11:1,12:1,13:3,14:3,15:1,16:1,17:1,18:1,19:5,
     20:1,21:1,22:1,23:1,24:3,25:2,26:3,27:1,28:5,29:2,30:1,31:1,32:2,33:3,34:1,35:1,36:1,
     37:1,38:1,39:1,40:1,41:1,42:1,43:1,44:1,45:1,46:5,47:3,48:2,49:1,50:2,51:1,52:1,53:1,
     54:1,55:2,56:1,57:1,58:1,59:1,60:1,61:1,62:1,63:1,64:1,65:1,66:2,67:5,68:1,69:2,70:3,
     71:2,72:1,73:1,74:1,75:1,76:1,77:2,78:2,79:5,80:1}
assert len(E) == 80 and len(C) == 80
RUB = {1:"leaves standing", 2:"says WRONG, sticks", 3:"hedges", 4:"adds nothing",
       5:"self-undoing", 6:"degenerate"}
BAR = "=" * 96

ROWS = []
for n, v in KEY.items():
    n = int(n)
    ROWS.append(dict(n=n, side=v["side"], arm=v["arm"], cell=v["direction"], dose=v["dose"],
                     G=v["G"], Q=v["Q"], claude=C[n], emil=E[n]))

# ---------------------------------------------------------------- 1. inter-rater
print(BAR); print("1. TWO HUMAN READERS, SAME 80 ITEMS, SCORED INDEPENDENTLY AND BLIND")
print(BAR)
agree = sum(1 for r in ROWS if r["claude"] == r["emil"])
print(f"exact agreement on the six-category rubric: {agree}/80 = {agree/80:.3f}")
# the decision that matters is binary: did the model deny the claim it was handed?
den_c = {r["n"]: int(r["claude"] in (2, 5)) for r in ROWS}
den_e = {r["n"]: int(r["emil"] in (2, 5)) for r in ROWS}
agree2 = sum(1 for n in den_c if den_c[n] == den_e[n])
print(f"agreement on the binary that carries the argument (denies vs does not): "
      f"{agree2}/80 = {agree2/80:.3f}")
po = agree2/80
pc = sum((sum(den_c.values())/80)*(sum(den_e.values())/80) for _ in [0]) + \
     ((80-sum(den_c.values()))/80)*((80-sum(den_e.values()))/80)
print(f"Cohen's kappa on that binary: {(po-pc)/(1-pc):.3f}")
print("\nitems where the two readers differ:")
for r in ROWS:
    if r["claude"] != r["emil"]:
        print(f"  [{r['n']:>2}] {r['side']:>5}/{r['arm']:<8} {r['cell']:>16} {r['dose']:>6} | "
              f"Claude {r['claude']} ({RUB[r['claude']]}) vs Emiliano {r['emil']} ({RUB[r['emil']]})")

# ---------------------------------------------------------------- 2. the asymmetry
print("\n" + BAR); print("2. THE ASYMMETRY, MEASURED AGAIN IN A SECOND MODEL FAMILY")
print(BAR)
def block(sel, label):
    n = len(sel)
    if not n: return None
    g = np.mean([r["G"] for r in sel]); q = np.mean([r["Q"] for r in sel])
    hc = np.mean([int(r["claude"] == 2) for r in sel])
    he = np.mean([int(r["emil"] == 2) for r in sel])
    h = (hc + he) / 2
    print(f"  {label:<34} n={n:>3} | Scorer G {g:.3f} | Scorer Q {q:.3f} | "
          f"human (C/E) {hc:.3f}/{he:.3f}")
    if h > 0:
        print(f"  {'':<34}       inflation vs mean human: G {g/h:.2f}x   Q {q/h:.2f}x")
    else:
        print(f"  {'':<34}       human genuine retraction is ZERO; any scorer rate is "
              f"pure false positive")
    return dict(n=n, G=float(g), Q=float(q), human_claude=float(hc), human_emil=float(he))

SUM = {}
SUM["false_inject"] = block([r for r in ROWS if r["side"]=="false" and r["arm"]=="inject"],
                            "false side, injected")
SUM["true_inject"]  = block([r for r in ROWS if r["side"]=="true"  and r["arm"]=="inject"],
                            "true side, injected")
SUM["true_random"]  = block([r for r in ROWS if r["side"]=="true"  and r["arm"]=="random"],
                            "true side, dose-matched random")
SUM["true_base"]    = block([r for r in ROWS if r["side"]=="true"  and r["arm"]=="baseline"],
                            "true side, baseline")
SUM["false_random"] = block([r for r in ROWS if r["side"]=="false" and r["arm"]=="random"],
                            "false side, dose-matched random")
SUM["false_base"]   = block([r for r in ROWS if r["side"]=="false" and r["arm"]=="baseline"],
                            "false side, baseline")

# ---------------------------------------------------------------- 3. by direction and dose
print("\n" + BAR); print("3. TRUE SIDE UNDER INJECTION, BY DIRECTION AND DOSE")
print(BAR)
print(f"{'direction':>17} {'dose':>7} {'n':>3} | {'G':>5} {'Q':>5} | {'C cat2':>7} {'E cat2':>7} | "
      f"{'C cat5':>7} {'E cat5':>7}")
cells = defaultdict(list)
for r in ROWS:
    if r["side"]=="true" and r["arm"]=="inject": cells[(r["cell"], r["dose"])].append(r)
for k in sorted(cells):
    s = cells[k]; n=len(s)
    print(f"{k[0]:>17} {k[1]:>7} {n:>3} | {np.mean([x['G'] for x in s]):>5.2f} "
          f"{np.mean([x['Q'] for x in s]):>5.2f} | "
          f"{np.mean([int(x['claude']==2) for x in s]):>7.2f} "
          f"{np.mean([int(x['emil']==2) for x in s]):>7.2f} | "
          f"{np.mean([int(x['claude']==5) for x in s]):>7.2f} "
          f"{np.mean([int(x['emil']==5) for x in s]):>7.2f}")

# ---------------------------------------------------------------- 4. false positives
print("\n" + BAR); print("4. WHERE THE SCORERS FIRE AND NEITHER READER SAW A DENIAL")
print(BAR)
for lab, kk in [("Scorer G", "G"), ("Scorer Q", "Q")]:
    fp = [r for r in ROWS if r[kk]==1.0 and r["claude"]!=2 and r["emil"]!=2]
    tot = [r for r in ROWS if r[kk]==1.0]
    print(f"\n{lab}: fires on {len(tot)} items, {len(fp)} of them with no denial from either reader")
    for r in fp:
        print(f"   [{r['n']:>2}] {r['side']:>5}/{r['arm']:<8} {r['cell']:>16} {r['dose']:>6} | "
              f"C={RUB[r['claude']]}, E={RUB[r['emil']]}")

# ---------------------------------------------------------------- 5. cross-model
print("\n" + BAR); print("5. THE SAME MEASUREMENT IN TWO MODEL FAMILIES")
print(BAR)
ti, fi = SUM["true_inject"], SUM["false_inject"]
hm_t = (ti["human_claude"] + ti["human_emil"]) / 2
hm_f = (fi["human_claude"] + fi["human_emil"]) / 2
print(f"{'':>26} {'false side':>26} | {'true side':>26}")
print(f"{'gemma-2-9b-it':>26} {'scorer 0.938  human 0.938':>26} | {'scorer 0.469  human 0.078':>26}")
print(f"{'':>26} {'inflation 1.00x':>26} | {'inflation 6.01x':>26}")
gq = f"scorer 0.{int(round(fi['Q']*1000)):03d}  human 0.{int(round(hm_f*1000)):03d}"
tq = f"scorer 0.{int(round(ti['Q']*1000)):03d}  human 0.{int(round(hm_t*1000)):03d}"
print(f"{'Qwen2.5-7B (Scorer Q)':>26} {gq:>26} | {tq:>26}")
inf_f = fi["Q"]/hm_f if hm_f else float('inf')
print(f"{'':>26} {('inflation ' + format(inf_f, '.2f') + 'x'):>26} | "
      f"{('inflation infinite' if hm_t == 0 else 'inflation ' + format(ti['Q']/hm_t, '.2f') + 'x'):>26}")

out = dict(inter_rater=dict(exact=agree/80, binary=agree2/80, kappa=(po-pc)/(1-pc)),
           blocks=SUM, scorer_G=REC["scorer_G"], scorer_Q=REC["scorer_Q"],
           shared=REC.get("scorer_shared"), calibration_chosen=REC.get("P2"),
           per_cell={f"{k[0]}|{k[1]}": dict(n=len(v),
                     G=float(np.mean([x['G'] for x in v])), Q=float(np.mean([x['Q'] for x in v])),
                     claude2=float(np.mean([int(x['claude']==2) for x in v])),
                     emil2=float(np.mean([int(x['emil']==2) for x in v])))
                     for k, v in cells.items()})
json.dump(out, open(O + "arc25_crossanalysis.json", "w"), indent=1)
print("\nsaved arc25_crossanalysis.json")
