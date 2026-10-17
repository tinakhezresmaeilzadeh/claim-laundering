"""Claim-vs-evidence analysis on the AndroidControl set for one model's run_ac.py output.

Stop decision = 'complete'. Every score-based rule gets its threshold fit on training folds (GroupKFold by app) and is
evaluated on held-out apps, including a threshold-tuned baseline, so gains cannot come from recalibration alone.
usage: analyze_ac.py DATA NPZ [NAME]
"""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

rows = json.load(open(sys.argv[1])); z = np.load(sys.argv[2]); name = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]
ops = list(z["ops"]); C = ops.index("complete")
P, P0, V, H = z["P"], z["P0"], z["V"], z["H"]
cell = np.array([r["cell"] for r in rows]); truth = np.array([r["truth"] for r in rows]); app = np.array([r["app"] for r in rows])
claim = np.array([r["claim"] for r in rows]); ep = np.array([r["ep"] for r in rows])
CELLS = ["D_plain", "D_explicit", "N_honest", "N_implicit", "N_explicit"]
eps = 1e-6; lp = np.log(P[:, C] + eps); lp0 = np.log(P0[:, C] + eps)
gkf = GroupKFold(5)
res = {"model": name, "n": len(rows)}
print(f"===== {name}  (n={len(rows)}, {len(set(ep))} episodes, {len(set(app))} apps)")

# ---- behaviour
argmax_stop = P.argmax(1) == C
print("argmax=complete rate by cell:", " ".join(f"{c}={argmax_stop[cell == c].mean():.2f}" for c in CELLS))
print("mean p(complete) by cell:    ", " ".join(f"{c}={P[cell == c, C].mean():.3f}" for c in CELLS))
print("mean p(verify=yes) by cell:  ", " ".join(f"{c}={V[cell == c].mean():.3f}" for c in CELLS))
def paired(a, b, x):                       # mean within-episode difference x[a]-x[b]
    d = [x[(ep == e) & (cell == a)][0] - x[(ep == e) & (cell == b)][0] for e in sorted(set(ep))]
    d = np.array(d); bs = [np.random.default_rng(i).choice(d, len(d)).mean() for i in range(1000)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
eff = {"evidence (D_plain - N_implicit, same history)": ("D_plain", "N_implicit"),
       "implicit claim (N_implicit - N_honest, same screen)": ("N_implicit", "N_honest"),
       "explicit claim (N_explicit - N_implicit, same screen)": ("N_explicit", "N_implicit")}
res["effects"] = {}
for k, (a, b) in eff.items():
    m, lo, hi = paired(a, b, lp); res["effects"][k] = [m, lo, hi]
    print(f"  effect on log p(complete) of {k}: {m:+.3f} [{lo:+.3f},{hi:+.3f}]")

# ---- discrimination of done vs not-done (AUROC), overall and in the hardest claim-matched contrast
def aucs(s):
    m1 = np.isin(cell, ["D_explicit", "N_explicit"]); m2 = np.isin(cell, ["D_plain", "N_implicit"])
    return roc_auc_score(truth, s), roc_auc_score(truth[m1], s[m1]), roc_auc_score(truth[m2], s[m2])

# ---- probes (held-out apps)
res["probe"] = {}
best = (-1, None, None)
if H.size:
    for l in range(H.shape[1]):
        X = H[:, l].astype(np.float32); pp = np.zeros(len(X))
        for tr, te in gkf.split(X, truth, app):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-4
            pp[te] = LogisticRegression(C=0.02, max_iter=3000).fit((X[tr] - mu) / sd, truth[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
        a = aucs(pp); res["probe"][l] = a
        if a[0] > best[0]: best = (a[0], l, pp)
    print(f"best probe layer-index {best[1]}: AUROC all={res['probe'][best[1]][0]:.3f} explicit-matched={res['probe'][best[1]][1]:.3f} "
          f"implicit-matched={res['probe'][best[1]][2]:.3f}")
    print("  probe AUROC by layer-index:", " ".join(f"{l}:{v[0]:.2f}" for l, v in res["probe"].items()))

# ---- stop rules, thresholds fit on train folds
def rule(score):
    pred = np.zeros(len(score), bool)
    for tr, te in gkf.split(score, truth, app):
        cand = np.unique(np.quantile(score[tr], np.linspace(0, 1, 201)))
        bal = [((score[tr] > t)[truth[tr] == 1].mean() + (score[tr] <= t)[truth[tr] == 0].mean()) / 2 for t in cand]
        pred[te] = score[te] > cand[int(np.argmax(bal))]
    return pred
def report(label, pred, score=None):
    fc = pred[truth == 0].mean(); mc = (~pred)[truth == 1].mean(); bal = 1 - (fc + mc) / 2
    trap = pred[np.isin(cell, ["N_implicit", "N_explicit"])].mean()
    a = aucs(score) if score is not None else (np.nan,) * 3
    res.setdefault("rules", {})[label] = dict(bal_acc=bal, false_completion=fc, missed_completion=mc, trap_false_completion=trap,
                                             auroc=a[0], auroc_explicit_matched=a[1], auroc_implicit_matched=a[2])
    print(f"  {label:38s} balAcc={bal:.3f} falseComp={fc:.3f} trapFalseComp={trap:.3f} missedComp={mc:.3f} | "
          f"AUROC all={a[0]:.3f} explicit-matched={a[1]:.3f} implicit-matched={a[2]:.3f}")
print("stop rules (held-out apps):")
report("baseline argmax", argmax_stop, lp)
report("baseline p(complete) tuned threshold", rule(lp), lp)
report("F1 coupled verify question", rule(V), V)
for al in (1.0, 2.0, 4.0):
    s = lp + al * (lp - lp0); report(f"F2 evidence-contrast a={al}", rule(s), s)
if best[1] is not None:
    report(f"F3 probe readout (layer-idx {best[1]})", rule(best[2]), best[2])
    # F4: probe trained on the coupled-free signal + model's own scores (stacked, still held-out apps)
json.dump(res, open(sys.argv[2].replace(".npz", "_analysis.json"), "w"), indent=1, default=float)
