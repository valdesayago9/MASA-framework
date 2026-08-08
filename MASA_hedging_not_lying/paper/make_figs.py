import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.0,
    "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 220, "savefig.dpi": 220,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})
INK, RED, BLU, GRY, ORG, PUR = "#1a1a1a", "#c0392b", "#2471a3", "#95a5a6", "#c07a00", "#7d3c98"
W = 6.5
OUT = "figs/"
os.makedirs(OUT, exist_ok=True)

R = json.load(open("/mnt/user-data/uploads/arc23b.json"))
A = json.load(open("/mnt/user-data/uploads/arc23c_addendum.json"))
LAD = R["ladder_results"]
LADDER = [0.2, 0.3, 0.4, 0.6, 0.9]
NAMES = ["d_apollo", "d_persona_final", "d_content_span", "d_content_final"]
NICE = {"d_apollo": "persona contrast, span read\n(published RepE mask)",
        "d_persona_final": "persona contrast, final-token read",
        "d_content_span": "content contrast, span read",
        "d_content_final": "content contrast, final-token read"}
SHORT = {"d_apollo": "persona / span", "d_persona_final": "persona / final",
         "d_content_span": "content / span", "d_content_final": "content / final"}

# ---------------------------------------------------------------- FIG 1
fig, axes = plt.subplots(2, 2, figsize=(W, 4.7), sharey=True, sharex=True)
for ax, nm in zip(axes.ravel(), NAMES):
    rr = LAD[nm]["rungs"]
    cs = [c for c in LADDER if str(c) in rr]
    ax.plot(cs, [rr[str(c)]["false"]["inject_rate"] for c in cs], "o-", color=BLU, lw=1.6, ms=4,
            label="false claim, direction")
    ax.plot(cs, [rr[str(c)]["false"]["random_rate"] for c in cs], "o--", color=BLU, lw=1.0, ms=3,
            alpha=.45, label="false claim, random twin")
    ax.plot(cs, [rr[str(c)]["true"]["inject_rate"] for c in cs], "s-", color=RED, lw=1.6, ms=4,
            label="true claim, direction")
    ax.plot(cs, [rr[str(c)]["true"]["random_rate"] for c in cs], "s--", color=RED, lw=1.0, ms=3,
            alpha=.45, label="true claim, random twin")
    ax.set_title(NICE[nm], fontsize=7.8, color=INK, pad=5)
    ax.set_ylim(-.05, 1.22)
    ax.set_xlim(.13, .97)
    ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.grid(axis="y", lw=.4, color="#dddddd")
for ax in axes[1]:
    ax.set_xlabel("injection strength (multiple of mean residual norm)", fontsize=7.5)
for ax in axes[:, 0]:
    ax.set_ylabel("walk-back rate\n(lexical scorer)", fontsize=7.5)
axes[0, 0].legend(frameon=False, fontsize=6.3, loc="upper left", ncol=2, columnspacing=.8,
                  handlelength=1.6)
fig.tight_layout()
fig.savefig(OUT + "fig1_ladder.png")
plt.close(fig)

# ---------------------------------------------------------------- FIG 2
fine = A["fine_grid"]["d_content_final"]
cs = [0.22, 0.24, 0.26, 0.28]
loose_t = [fine[str(c)]["true"]["inject_rate"] for c in cs]
f_eff = [fine[str(c)]["false"]["loose"]["effect"] for c in cs]
f_lo = [fine[str(c)]["false"]["loose"]["ci"][0] for c in cs]
f_hi = [fine[str(c)]["false"]["loose"]["ci"][1] for c in cs]
hedge = [1 / 6, 2 / 6, 3 / 6, 4 / 6]

fig, ax = plt.subplots(figsize=(W, 3.5))
x = np.arange(4)
ax.bar(x - .27, f_eff, .23, color=BLU, label="false claims corrected, effect over random twin")
ax.errorbar(x - .27, f_eff,
            yerr=[np.array(f_eff) - np.array(f_lo), np.array(f_hi) - np.array(f_eff)],
            fmt="none", ecolor=INK, elinewidth=.8, capsize=2.5)
ax.bar(x, loose_t, .23, color=GRY, label="true claims, what the lexical scorer calls retraction")
ax.bar(x + .27, hedge, .23, color=ORG, label="true claims, hedging under blind human audit")
ax.plot(x + .27, [0, 0, 0, 0], "v", color=RED, ms=8, clip_on=False,
        label="true claims, genuine retraction under blind human audit = 0")
for xi, h in zip(x, hedge):
    ax.text(xi + .27, h + .025, format(h, ".2f"), ha="center", fontsize=7, color=ORG)
for xi, g in zip(x, loose_t):
    ax.text(xi, g + .025, format(g, ".2f"), ha="center", fontsize=7, color="#6d7b7b")
ax.set_xticks(x)
ax.set_xticklabels(["c = " + str(c) for c in cs], fontsize=8.5)
ax.set_ylabel("rate")
ax.set_ylim(-.03, 1.32)
ax.set_yticks([0, .25, .5, .75, 1.0])
ax.grid(axis="y", lw=.4, color="#dddddd")
ax.legend(frameon=False, fontsize=6.8, loc="upper left", handlelength=1.4)
fig.tight_layout()
fig.savefig(OUT + "fig2_window.png")
plt.close(fig)

# ---------------------------------------------------------------- FIG 3
cells = [("Arc 22 operating point", 0.335, 1 / 8, 1 / 8, 1 / 8),
         ("content / final,  c = 0.3", LAD["d_content_final"]["rungs"]["0.3"]["true"]["inject_rate"], 0.0, 3 / 4, 0.0),
         ("content / final,  c = 0.4", LAD["d_content_final"]["rungs"]["0.4"]["true"]["inject_rate"], 1 / 4, 0.0, 1 / 4),
         ("content / final,  c = 0.6", LAD["d_content_final"]["rungs"]["0.6"]["true"]["inject_rate"], 3 / 4, 0.0, 1 / 4),
         ("content / final,  c = 0.9", LAD["d_content_final"]["rungs"]["0.9"]["true"]["inject_rate"], 0.0, 0.0, 1.0),
         ("content / span,  c = 0.4", LAD["d_content_span"]["rungs"]["0.4"]["true"]["inject_rate"], 0.0, 0.0, 0.5),
         ("content / span,  c = 0.6", LAD["d_content_span"]["rungs"]["0.6"]["true"]["inject_rate"], 0.0, 0.0, 1.0),
         ("persona / final,  c = 0.9", LAD["d_persona_final"]["rungs"]["0.9"]["true"]["inject_rate"], 0.0, 0.0, 0.0)]
y = np.arange(len(cells))[::-1]
fig, ax = plt.subplots(figsize=(W, 3.9))
ax.barh(y + .19, [c[1] for c in cells], .34, color=GRY, label="lexical scorer, all 24 items")
left = np.zeros(len(cells))
for val, col, name in [([c[2] for c in cells], RED, "human: genuine retraction"),
                       ([c[3] for c in cells], ORG, "human: hedging"),
                       ([c[4] for c in cells], PUR, "human: self-undoing loop")]:
    ax.barh(y - .19, val, .34, left=left, color=col, label=name)
    left = left + np.array(val)
ax.set_yticks(y)
ax.set_yticklabels([c[0] for c in cells], fontsize=7.6)
ax.set_xlabel("rate on true prefilled claims", fontsize=7.8)
ax.set_xlim(0, 1.02)
ax.set_ylim(-.9, len(cells) - .1)
ax.grid(axis="x", lw=.4, color="#dddddd")
ax.legend(frameon=False, fontsize=6.9, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2,
          handlelength=1.4)
fig.tight_layout()
fig.savefig(OUT + "fig3_scorer_vs_human.png")
plt.close(fig)

# ---------------------------------------------------------------- FIG 4
up = A["undo_profile"]
fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.05, 1]})
ax = axes[0]
for nm, col in [("d_content_final", RED), ("d_content_span", PUR),
                ("d_persona_final", BLU), ("d_apollo", GRY)]:
    xs = [c for c in LADDER if nm + "|true|inject|c" + str(c) in up]
    ax.plot(xs, [up[nm + "|true|inject|c" + str(c)]["attractor"] for c in xs], "o-", color=col,
            lw=1.5, ms=4, label=SHORT[nm])
ax.set_xlabel("injection strength", fontsize=7.8)
ax.set_ylabel("self-undoing loop rate", fontsize=7.8)
ax.set_title("with a claim prefilled", fontsize=8.2, color=INK)
ax.set_ylim(-.04, 1.08)
ax.set_yticks([0, .25, .5, .75, 1.0])
ax.grid(axis="y", lw=.4, color="#dddddd")
ax.legend(frameon=False, fontsize=6.6, loc="upper left", ncol=2, columnspacing=.7, handlelength=1.4)

ax = axes[1]
free = {"baseline": 0.0, "content/final\nc=0.4": 0.0, "content/final\nc=0.6": 0.0,
        "content/span\nc=0.4": 0.0, "content/span\nc=0.6": 0.0, "persona/final\nc=0.6": 0.25}
ax.bar(range(len(free)), list(free.values()), .55, color=INK)
ax.set_xticks(range(len(free)))
ax.set_xticklabels(list(free), fontsize=6.4)
ax.set_ylabel("self-contradiction rate", fontsize=7.8)
ax.set_ylim(-.04, 1.08)
ax.set_yticks([0, .25, .5, .75, 1.0])
ax.set_title("same injection, free generation", fontsize=8.2, color=INK)
ax.grid(axis="y", lw=.4, color="#dddddd")
fig.tight_layout()
fig.savefig(OUT + "fig4_prefill.png")
plt.close(fig)

# ---------------------------------------------------------------- FIG 5
single = {"d_apollo": .168, "d_persona_final": .565, "d_content_span": .078, "d_content_final": .703}
rep = {k: (R["probe"][k]["floor"], R["probe"][k]["floor_sd"]) for k in NAMES}
fig, ax = plt.subplots(figsize=(W, 3.0))
x = np.arange(4)
ax.bar(x - .19, [single[n] for n in NAMES], .34, color=GRY,
       label="one fixed split, layer chosen by widest margin")
ax.bar(x + .19, [rep[n][0] for n in NAMES], .34, color=BLU, label="25 repeated splits")
ax.errorbar(x + .19, [rep[n][0] for n in NAMES], yerr=[rep[n][1] for n in NAMES],
            fmt="none", ecolor=INK, elinewidth=.8, capsize=3)
ax.axhline(.5, color=RED, lw=1.0, ls=":", label="chance")
ax.set_xticks(x)
ax.set_xticklabels([SHORT[n] for n in NAMES], fontsize=8)
ax.set_ylabel("permutation floor (AUROC)", fontsize=7.8)
ax.set_ylim(0, 1.24)
ax.set_yticks([0, .25, .5, .75, 1.0])
ax.grid(axis="y", lw=.4, color="#dddddd")
ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=2, columnspacing=.9, handlelength=1.4)
fig.tight_layout()
fig.savefig(OUT + "fig5_floors.png")
plt.close(fig)

from PIL import Image
print("figure                        pixels        inches at 220 dpi")
for f in sorted(os.listdir(OUT)):
    im = Image.open(OUT + f)
    print("  " + f.ljust(28) + str(im.size[0]) + " x " + str(im.size[1]) + "   "
          + format(im.size[0] / 220, ".2f") + " x " + format(im.size[1] / 220, ".2f"))
