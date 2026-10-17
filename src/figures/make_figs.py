"""Paper figures (single-column PDFs). Palette: validated categorical slots 1-4 (dataviz reference palette); every
series is direct-labelled and has its own marker, because slots 3-4 are below 3:1 contrast on white."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                     "lines.linewidth": 1.4, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
R = "results/"
MODELS = [("decider4b_sub", "decider-4b", "o", lambda k: 2 * k), ("kev4b", "Kev-4B", "s", lambda k: 2 * (k + 1)),
          ("qwenchat4b", "Qwen3.5-4B (chat)", "^", lambda k: 2 * k), ("qwenbase4b", "Qwen3.5-4B-Base", "D", lambda k: 2 * k)]

# ---- Fig. 1: what the model knows vs what it outputs. (a) phones: probe AUROC by relative depth, output heads as a band;
# (b) all domains: output head (hollow) -> nested probe (filled), text-only classifier as a grey tick. Same models, same colours.
LC = json.load(open(R + "layer_curves.json"))      # src/figures/layer_curves.py
DEC = [("decider-4b", "decider4b", "o"), ("Kev-4B", "kev4b_full", "s"), ("decider-12b", "decider12b", "D")]
fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.3, 2.05), sharey=True, gridspec_kw=dict(width_ratios=[1.1, 1]))
heads = [LC["phones"][m]["head"] for m, _, _ in DEC]
ax.axhspan(min(heads), max(heads), color=GRID, lw=0, zorder=0)
ax.text(98, (min(heads) + max(heads)) / 2, "output heads", fontsize=6, color=INK2, va="center", ha="right")
for (m, _, mk), col in zip(DEC, C):
    d = LC["phones"][m]; x = [100 * v for v in d["depth"]]
    ax.plot(x, d["auroc"], color=col, marker=mk, ms=2.6, lw=1.3, mec="white", mew=0.4, zorder=3, label=m)
ax.legend(loc="upper left", fontsize=6, frameon=False, handlelength=1.6, borderaxespad=0.2, labelspacing=0.3)
ax.axhline(0.5, color=INK2, lw=0.6)
ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xticklabels(["0", "50", "100%"])
ax.set_xlabel("Layer depth (phones)"); ax.set_ylabel("Done vs. not-done AUROC"); ax.set_ylim(0.45, 1.03)
ax.set_title("(a) Where it is encoded", fontsize=7, loc="left", color=INK)
FIN = {"phones": lambda f: json.load(open(R + f"ac_{f}_final.json"))["auroc"]}
TXT = {"Phones": 0.594, "ALFWorld": 1.0, "WebShop": 1.0}
for gi, (dom, key) in enumerate([("Phones", "phones"), ("ALFWorld", "alf_eval"), ("WebShop", "ws_eval")]):
    for mi, ((m, f, mk), col) in enumerate(zip(DEC, C)):
        if key == "phones":
            a = json.load(open(R + f"ac_{f}_final.json"))["auroc"]; h, p = a["output head p(complete)"]["all"], a["nested probe"]["all"]
        else:
            a = json.load(open(R + f"{key}_{f.replace('_full', '')}_analysis.json"))["auroc"]; h, p = a["head proceed-vs-retry"]["all"], a["nested probe"]["all"]
        x = gi + (mi - 1) * 0.24
        bx.plot([x, x], [h, p], color=col, lw=1.3, zorder=2)
        bx.plot([x], [h], marker=mk, ms=3.6, mfc="white", mec=col, mew=1.0, zorder=3)
        bx.plot([x], [p], marker=mk, ms=3.6, color=col, mec="white", mew=0.4, zorder=3)
    bx.plot([gi - 0.38, gi + 0.38], [TXT[dom]] * 2, color=INK2, lw=0.9, ls=(0, (2, 1.5)), zorder=1)
bx.axhline(0.5, color=INK2, lw=0.6)
bx.set_xticks(range(3)); bx.set_xticklabels(["Phones", "ALFWorld", "WebShop"], fontsize=6); bx.set_xlim(-0.5, 2.5)
bx.tick_params(axis="x", length=0); bx.grid(axis="x", visible=False)
bx.set_title("(b) Head $\\circ$ vs. probe $\\bullet$", fontsize=7, loc="left", color=INK)
fig.tight_layout(pad=0.3, w_pad=0.5); fig.savefig(OUT + "/probe_by_layer.pdf"); plt.close(fig)

# ---- Fig. 2: label efficiency (nested-layer probe, held-out apps)
fig, ax = plt.subplots(figsize=(3.3, 1.9))
for (f, name, mk, _), col in zip(MODELS, C):
    lc = json.load(open(R + f"ac_{f}_final.json"))["learning_curve"]; xs = sorted(int(k) for k in lc)
    head = json.load(open(R + f"ac_{f}_final.json"))["auroc"]["output head p(complete)"]["all"]
    ax.plot(xs, [lc[str(k)] for k in xs], color=col, marker=mk, ms=3)
    ax.annotate(name, (xs[-1], lc[str(xs[-1])]), xytext=(3, 0), textcoords="offset points", va="center", fontsize=6.5)
    ax.plot([xs[0]], [head], marker="x", color=col, ms=4, mew=1)
ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100, 200]); ax.set_xticklabels(["10", "25", "50", "100", "200"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlim(8, 520); ax.set_xlabel("Labelled training episodes"); ax.set_ylabel("Done vs. not-done AUROC")
ax.text(9, 0.772, "lines: probe    ×: output head", fontsize=6, color=INK2)
fig.tight_layout(pad=0.3); fig.savefig(OUT + "/learning_curve.pdf"); plt.close(fig)

# ---- Fig. 3: closed loop, paired change in success vs cost (24 tasks x 30 seeds)
e = json.load(open(R + "e2e_pooled_s30.json"))
names = {"none": ("No gate", (6, -9)), "random": ("Random (matched)", (6, 3)), "contrast": ("Evidence contrast", (-6, 4)),
         "verify": ("Verify question", (6, 3)), "probe": ("EviGate", (-6, 4))}
fig, ax = plt.subplots(figsize=(3.3, 1.9))
ax.axhline(0, color=INK2, lw=0.6)
for g, (lab, off) in names.items():
    r = e[g]; col = C[0] if g == "probe" else INK2; d = 100 * r["delta"]; lo, hi = 100 * r["delta_ci"][0], 100 * r["delta_ci"][1]
    ax.errorbar(r["calls_per_success"], d, yerr=[[d - lo], [hi - d]] if g != "none" else None, fmt="o", color=col, ms=4, elinewidth=0.9, capsize=0)
    ax.annotate(lab, (r["calls_per_success"], d), xytext=off, textcoords="offset points", ha="left" if off[0] > 0 else "right",
                fontsize=6.5, color=INK, fontweight="bold" if g == "probe" else "normal")
ax.set_xlabel("Model calls per successful task"); ax.set_ylabel("Δ success vs. no gate (pp)")
ax.set_xlim(3.6, 12.6); ax.set_ylim(-11, 13)
fig.tight_layout(pad=0.3); fig.savefig(OUT + "/e2e_cost.pdf"); plt.close(fig)
# ---- Fig. 4: mechanism, answer-slot patching restoration by layer (decider-4b)
mz = json.load(open(R + "mechanism_patching.json")); Ls = mz["layers"]
fig, ax = plt.subplots(figsize=(3.3, 1.9))
for (k, lab, mk), col in zip([("evidence", "Screen evidence", "o"), ("implicit", "Implicit claim", "s"), ("claim", "Explicit claim", "^")], C):
    mu = np.array(mz[k]["mean"]); se = np.array(mz[k]["se"])
    ax.fill_between(Ls, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.15, lw=0)
    ax.plot(Ls, mu, color=col, marker=mk, ms=3, label=f"{lab} (n={mz[k]['n']})")
ax.legend(loc="upper left", frameon=False, fontsize=6.5, handlelength=1.8)
ax.set_xlabel("Layer patched at the answer slot (of 32)"); ax.set_ylabel("Fraction of effect restored")
ax.set_xticks(range(0, 33, 8)); ax.set_ylim(-0.05, 1.05)
fig.tight_layout(pad=0.3); fig.savefig(OUT + "/mechanism.pdf"); plt.close(fig)
# ---- Fig. 5: mechanism in three domains (decider-4b), one panel per domain
fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.85), sharey=True)
for ax, (f, title) in zip(axes, [("mechanism_patching.json", "Phones"), ("mechanism_alf.json", "ALFWorld"), ("mechanism_ws.json", "WebShop")]):
    mz = json.load(open(R + f)); Ls = mz["layers"]
    for (k, lab, mk), col in zip([("evidence", "Observation", "o"), ("implicit", "Logged failed step", "s"), ("claim", "Explicit claim", "^")], C):
        if not mz.get(k) or mz[k]["n"] < 20: continue
        mu = np.array(mz[k]["mean"]); se = np.array(mz[k]["se"])
        ax.fill_between(Ls, mu - 1.96 * se, mu + 1.96 * se, color=col, alpha=0.15, lw=0)
        ax.plot(Ls, mu, color=col, marker=mk, ms=2.5, label=lab)
    ax.set_title(title, fontsize=7.5); ax.set_xticks(range(0, 33, 8)); ax.set_ylim(-0.08, 1.05); ax.set_xlabel("Layer patched")
axes[0].set_ylabel("Effect restored"); axes[0].legend(loc="upper left", frameon=False, fontsize=6, handlelength=1.5)
fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(OUT + "/mechanism3.pdf"); plt.close(fig)
print("ok")
