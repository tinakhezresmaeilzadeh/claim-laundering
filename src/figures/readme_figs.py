"""README graphics, light and dark versions: the banner with the headline numbers, the input effects (Table 2, decider-4b)
and live-agent success (Table 5). Values are the paper's; palette = validated categorical slots 1-3.
usage: readme_figs.py OUTDIR
"""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "assets"
THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", card="#f3f2ee",
                  s=["#2a78d6", "#eb6834", "#1baf7a"], neutral="#a9a79f", neutral2="#d3d1ca"),
    "dark": dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#383835", card="#262624",
                 s=["#3987e5", "#d95926", "#199e70"], neutral="#76746c", neutral2="#4d4c47"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for k in ("top", "right", "left"): ax.spines[k].set_visible(False)
    ax.spines["bottom"].set_color(t["ink2"])
    ax.tick_params(colors=t["ink2"], length=0)
    ax.yaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)


def banner(t, name):
    fig = plt.figure(figsize=(12, 3.4), dpi=150, facecolor=t["bg"]); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.4)
    ax.text(0.45, 2.85, "Claim Laundering", color=t["ink"], fontsize=26, fontweight="bold", va="center")
    ax.text(0.45, 2.33, "When agent decision models trust the agent's record over what the environment shows",
            color=t["ink2"], fontsize=13, va="center")
    tiles = [("53–100%", "of failed steps that claim success\nstill end in stop / proceed / buy", t["s"][1]),
             ("0.77–1.00", "AUROC of a linear probe on the hidden\nstate: the model knows the step failed", t["s"][0]),
             ("+5 to +19", "points of live task success with\nEviGate, a one-dot-product veto", t["s"][2])]
    for i, (num, txt, c) in enumerate(tiles):
        x = 0.45 + i * 3.8
        ax.add_patch(FancyBboxPatch((x, 0.3), 3.55, 1.6, boxstyle="round,pad=0,rounding_size=0.12", fc=t["card"], ec="none"))
        ax.add_patch(FancyBboxPatch((x, 0.3), 0.07, 1.6, boxstyle="square,pad=0", fc=c, ec="none"))
        ax.text(x + 0.3, 1.48, num, color=t["ink"], fontsize=22, fontweight="bold", va="center")
        ax.text(x + 0.3, 0.88, txt, color=t["ink2"], fontsize=10.5, va="center", linespacing=1.35)
    fig.savefig(f"{OUT}/banner_{name}.png", facecolor=t["bg"]); plt.close(fig)


def effects(t, name):
    # Table 2, decider-4b: paired effect on log p(terminal option) of each input
    doms = ["Phones", "ALFWorld", "WebShop"]
    vals = {"Observation": [0.05, 1.19, 0.02], "Logged step": [1.10, 2.60, 3.71], "Success claim": [0.80, 1.28, 0.07]}
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=150, facecolor=t["bg"]); style(ax, t)
    w = 0.26
    for j, (lab, v) in enumerate(vals.items()):
        xs = [i + (j - 1) * (w + 0.02) for i in range(3)]
        ax.bar(xs, v, w, color=t["s"][j], label=lab, zorder=3, edgecolor=t["bg"], linewidth=1.5)
        for x, y in zip(xs, v): ax.text(x, y + 0.06, f"{y:+.2f}", ha="center", va="bottom", fontsize=8.5, color=t["ink2"])
    ax.set_xticks(range(3)); ax.set_xticklabels(doms, color=t["ink"], fontsize=11)
    ax.set_ylabel("Effect on log p(stop / proceed / buy)", color=t["ink2"], fontsize=10)
    ax.set_ylim(0, 4.2)
    ax.set_title("The record moves the decision more than the observation (decider-4b)", color=t["ink"], fontsize=11.5, loc="left", pad=12)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9.5, ncol=3, bbox_to_anchor=(0, 1.0))
    for tx in leg.get_texts(): tx.set_color(t["ink2"])
    fig.tight_layout(); fig.savefig(f"{OUT}/effects_{name}.png", facecolor=t["bg"]); plt.close(fig)


def live(t, name):
    # Table 5: task success (%) of live agents; random veto at a matched rate
    doms = ["Phone simulator", "ALFWorld", "WebShop"]
    rows = [("No gate", [77.2, 42.5, 8.5], t["neutral2"]), ("Random veto (matched rate)", [77.6, 58.2, 4.5], t["neutral"]),
            ("EviGate", [82.4, 61.2, 13.5], t["s"][0])]
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.6), dpi=150, facecolor=t["bg"])
    for i, ax in enumerate(axs):
        style(ax, t)
        v = [r[1][i] for r in rows]
        ax.bar(range(3), v, 0.72, color=[r[2] for r in rows], zorder=3, edgecolor=t["bg"], linewidth=1.5)
        for x, y in enumerate(v):
            ax.text(x, y + max(v) * 0.03, f"{y:.1f}", ha="center", va="bottom", fontsize=9.5,
                    color=t["ink"] if x == 2 else t["ink2"], fontweight="bold" if x == 2 else "normal")
        ax.set_xticks([]); ax.set_ylim(0, max(v) * 1.22)
        ax.set_title(doms[i], color=t["ink"], fontsize=11)
    axs[0].set_ylabel("Task success (%)", color=t["ink2"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=r[2]) for r in rows]
    leg = fig.legend(handles, [r[0] for r in rows], loc="lower center", ncol=3, frameon=False, fontsize=10)
    for tx in leg.get_texts(): tx.set_color(t["ink2"])
    fig.suptitle("EviGate raises live task success in every domain", color=t["ink"], fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.savefig(f"{OUT}/live_{name}.png", facecolor=t["bg"]); plt.close(fig)


for name, t in THEMES.items():
    banner(t, name); effects(t, name); live(t, name)
print("wrote banner/effects/live _light/_dark.png to", OUT)
