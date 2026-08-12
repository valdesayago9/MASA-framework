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
INK, RED, BLU, GRY, ORG, PUR, GRN = ("#1a1a1a", "#c0392b", "#2471a3", "#95a5a6",
                                     "#c07a00", "#7d3c98", "#1e8449")
W = 6.5
OUT = "figs/"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- FIG 1: the two failure modes
fig, axes = plt.subplots(1, 2, figsize=(W, 3.2))

ax = axes[0]
x = np.arange(2)
sc = [0.938, 0.469]
hu = [0.938, 0.078]
ax.bar(x - .19, sc, .34, color=GRY, label="marker scorer")
ax.bar(x + .19, hu, .34, color=RED, label="blind human audit")
for xi, (a, b) in enumerate(zip(sc, hu)):
    ax.text(xi - .19, a + .02, format(a, ".3f"), ha="center", fontsize=7, color="#5b6a6a")
    ax.text(xi + .19, b + .02, format(b, ".3f"), ha="center", fontsize=7, color=RED)
ax.text(0, 1.02, "1.0x", ha="center", fontsize=9.5, color=INK, fontweight="bold")
ax.text(1, 0.56, "6.0x", ha="center", fontsize=9.5, color=RED, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["false claims\nn=16", "true claims\nn=64"], fontsize=8)
ax.set_ylabel("retraction rate")
ax.set_ylim(0, 1.30)
ax.set_yticks([0, .25, .5, .75, 1.0])
ax.set_title("under injection: counts what is not there", fontsize=8.4, color=INK, pad=6)
ax.grid(axis="y", lw=.4, color="#dddddd")
ax.legend(frameon=False, fontsize=6.9, loc="upper center", bbox_to_anchor=(.5, -0.30), ncol=2)

ax = axes[1]
seg = [0.359, 0.438, 0.203]
cols = [GRY, RED, GRN]
labs = ["neutral", "defends the falsehood", "corrects, unseen"]
left = 0.0
for v, c, l in zip(seg, cols, labs):
    ax.barh([0], [v], 0.42, left=left, color=c, label=l)
    ax.text(left + v / 2, 0, format(v, ".3f"), ha="center", va="center", fontsize=8,
            color="white", fontweight="bold")
    left += v
ax.set_yticks([])
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.32, 0.32)
ax.spines["left"].set_visible(False)
ax.set_xlabel("share of the region no scorer sees (n=64 audited)", fontsize=7.8)
ax.set_title("under prefill: misses what is there", fontsize=8.4, color=INK, pad=6)
ax.grid(axis="x", lw=.4, color="#dddddd")
ax.legend(frameon=False, fontsize=6.9, loc="upper center", bbox_to_anchor=(.5, -0.30), ncol=3)
fig.tight_layout()
fig.savefig(OUT + "fig1_two_modes.png")
plt.close(fig)

# ---------------------------------------------------------------- FIG 2: per-marker
M = [("note that this", 5, 0, 0, 5, 0), ("however", 8, 2, 0, 5, 1), ("in fact", 2, 0, 1, 0, 0),
     ("in reality", 5, 2, 3, 0, 0), ("this is incorrect", 4, 2, 2, 0, 0), ("mistake", 2, 1, 1, 0, 0),
     ("actually", 25, 13, 10, 1, 1), ("just kidding", 24, 12, 12, 0, 0)]
M = sorted(M, key=lambda r: ((r[2] + r[3]) / r[1], r[1]))
fig, ax = plt.subplots(figsize=(W, 3.2))
y = np.arange(len(M))[::-1]
left = np.zeros(len(M))
for idx, col, lab in [(2, RED, "genuine retraction"), (3, PUR, "self-undoing loop"),
                      (4, ORG, "hedge"), (5, BLU, "leaves the claim standing")]:
    v = np.array([r[idx] / r[1] for r in M])
    ax.barh(y, v, .60, left=left, color=col, label=lab)
    left = left + v
for yi, r in zip(y, M):
    ax.text(1.02, yi, "n=" + str(r[1]), va="center", fontsize=7, color="#666666")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in M], fontsize=8)
ax.set_xlim(0, 1.13)
ax.set_xticks([0, .25, .5, .75, 1.0])
ax.set_xlabel("what a blind reader called the items each marker fired on", fontsize=7.8)
ax.grid(axis="x", lw=.4, color="#dddddd")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -0.16), ncol=4)
fig.tight_layout()
fig.savefig(OUT + "fig2_markers.png")
plt.close(fig)

# ---------------------------------------------------------------- FIG 3: retuning is worse
fig, ax = plt.subplots(figsize=(W, 3.0))
x = np.arange(2)
imported = [0.929, 0.190]
custom = [0.643, 0.286]
human = [0.714, 0.000]
ax.bar(x - .26, imported, .24, color=BLU, label="imported list, built on another model")
ax.bar(x, custom, .24, color=ORG, label="custom list, mined from this model's own false side")
ax.bar(x + .26, human, .24, color=RED, label="blind human audit, two readers")
for xi in x:
    for off, v, c in [(-.26, imported[xi], BLU), (0, custom[xi], ORG), (.26, human[xi], RED)]:
        ax.text(xi + off, v + .018, format(v, ".3f"), ha="center", fontsize=7, color=c)
ax.annotate("", xy=(1.0, 0.33), xytext=(0.74, 0.235),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
ax.text(1.02, 0.355, "retuning moves it\nfurther from the truth", fontsize=7.4, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(["false claims  n=14\nwhere the behaviour is expected",
                    "true claims  n=42\nwhere specificity is claimed"], fontsize=8)
ax.set_ylabel("retraction rate")
ax.set_ylim(0, 1.16)
ax.set_yticks([0, .25, .5, .75, 1.0])
ax.grid(axis="y", lw=.4, color="#dddddd")
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(OUT + "fig3_retuning.png")
plt.close(fig)

# ---------------------------------------------------------------- FIG 4: the blind region
fig, ax = plt.subplots(figsize=(W, 3.3))
stages = ["false-side\nitems", "no scorer\nfires", "coherent text\n(sampled)", "active defence\n(estimated)"]
vals = [1152, 418, 325, 142]
cols = [GRY, "#7f8c8d", BLU, RED]
bars = ax.bar(range(4), vals, .55, color=cols)
for i, (v, b) in enumerate(zip(vals, bars)):
    ax.text(i, v + 22, str(v), ha="center", fontsize=8.5, color=INK, fontweight="bold")
    if i:
        ax.text(i, v / 2, format(v / 1152, ".3f") + "\nof false side", ha="center", va="center",
                fontsize=7.2, color="white")
ax.set_xticks(range(4))
ax.set_xticklabels(stages, fontsize=7.8)
ax.set_ylabel("items")
ax.set_ylim(0, 1330)
ax.grid(axis="y", lw=.4, color="#dddddd")
ax.set_title("from the corpus to the failure no instrument reports", fontsize=8.6,
             color=INK, pad=6, loc="left")
fig.tight_layout()
fig.savefig(OUT + "fig4_blind_region.png")
plt.close(fig)

# ---------------------------------------------------------------- FIG 5: who defends
fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
models = ["llama-3.1-8b", "llama-3.3-70b", "claude-sonnet-4.5"]
defend = [0.500, 0.417, 0.375]
corr = [0.167, 0.042, 0.500]
x = np.arange(3)
ax.bar(x - .19, defend, .34, color=RED, label="defends the falsehood")
ax.bar(x + .19, corr, .34, color=GRN, label="corrects, and the scorer misses it")
for xi in x:
    ax.text(xi - .19, defend[xi] + .015, format(defend[xi], ".2f"), ha="center", fontsize=7, color=RED)
    ax.text(xi + .19, corr[xi] + .015, format(corr[xi], ".2f"), ha="center", fontsize=7, color=GRN)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=7.4, rotation=12, ha="right")
ax.set_ylabel("share of the blind region")
ax.set_ylim(0, 0.78)
ax.grid(axis="y", lw=.4, color="#dddddd")
ax.legend(frameon=False, fontsize=6.9, loc="upper center", ncol=1)

ax = axes[1]
pools = ["widely known\nfacts  n=53", "obscure\nfacts  n=11"]
pv = [0.377, 0.727]
ax.bar([0, 1], pv, .5, color=[GRY, RED])
for i, v in enumerate(pv):
    ax.text(i, v + .02, format(v, ".3f"), ha="center", fontsize=8.5, color=INK)
ax.set_xticks([0, 1])
ax.set_xticklabels(pools, fontsize=7.6)
ax.set_ylabel("defends the falsehood")
ax.set_ylim(0, 0.95)
ax.grid(axis="y", lw=.4, color="#dddddd")
ax.set_title("defence rises where the model is least\nlikely to know the answer", fontsize=7.8,
             color=INK, pad=5)
fig.tight_layout()
fig.savefig(OUT + "fig5_who_defends.png")
plt.close(fig)

from PIL import Image
for f in sorted(os.listdir(OUT)):
    print(f.ljust(28), Image.open(OUT + f).size)
