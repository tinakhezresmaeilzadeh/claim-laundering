"""Behavioural claim-vs-observation analysis of hosted Jev (run_jev.py output), same metrics as the open models minus probes.
usage: analyze_jev.py {ac|alf|ws} DATA JSONL
"""
import json, sys, numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

KIND, DATA, JL = sys.argv[1:4]
rows = json.load(open(DATA)); recs = {r["i"]: r for r in map(json.loads, open(JL))}
idx = [i for i in range(len(rows)) if i in recs and recs[i]["op"] and recs[i]["ver"]]
rows = [rows[i] for i in idx]; R = [recs[i] for i in idx]; EPS = 1e-4
MAP = {"S_plain": "D_plain", "S_think": "D_explicit", "F_honest": "N_honest", "F_implicit": "N_implicit", "F_explicit": "N_explicit"}
if KIND == "ac":
    pkey = lambda r, x: "complete"
    grp = np.array([r["app"] for r in rows]); key = np.array([str(r["ep"]) for r in rows])
else:
    pkey = lambda r, x: f"A{r['proceed']}"
    grp = np.array([r.get("game", r.get("goal")) for r in rows]); key = np.array([f"{r.get('game', r.get('goal'))}_{r.get('step', 0)}" for r in rows])
cell = np.array([MAP.get(r["cell"], r["cell"]) for r in rows]); truth = np.array([r["truth"] for r in rows])
pp = np.array([x["op"]["probabilities"].get(pkey(r, x), 0.0) for r, x in zip(rows, R)])
lp = np.log(np.clip(pp, EPS, 1)); V = np.array([x["ver"]["noul"] for x in R])
choice = np.array([x["op"]["choice"] == pkey(r, x) for r, x in zip(rows, R)])
out = {"n": len(rows), "cost": sum(x["cost"] for x in R)}
CELLS = ["D_plain", "D_explicit", "N_honest", "N_implicit", "N_explicit"]
out["stop_or_proceed_rate"] = {c: float(choice[cell == c].mean()) for c in CELLS}
def paired(a, b):
    d = np.array([lp[(key == k) & (cell == a)][0] - lp[(key == k) & (cell == b)][0] for k in sorted(set(key)) if ((key == k) & (cell == a)).any() and ((key == k) & (cell == b)).any()])
    bs = [np.random.default_rng(i).choice(d, len(d)).mean() for i in range(1000)]
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out["effects"] = {"evidence": paired("D_plain", "N_implicit"), "implicit_claim": paired("N_implicit", "N_honest"), "explicit_claim": paired("N_explicit", "N_implicit")}
m = cell != "N_honest" if KIND != "ac" else np.ones(len(rows), bool)
EXM = np.isin(cell, ["D_explicit", "N_explicit"])
def auc(s, mm): return float(roc_auc_score(truth[mm], s[mm])) if len(set(truth[mm])) > 1 else float("nan")
out["auroc"] = {"head": [auc(lp, m), auc(lp, m & EXM)], "verify": [auc(V, m), auc(V, m & EXM)]}
def rule(s):
    y, g, ss = truth[m], grp[m], s[m]; pred = np.zeros(len(ss), bool)
    for tr, te in GroupKFold(5).split(ss, y, g):
        c = np.unique(np.quantile(ss[tr], np.linspace(0, 1, 201)))
        bal = [((ss[tr] > t)[y[tr] == 1].mean() + (ss[tr] <= t)[y[tr] == 0].mean()) / 2 for t in c]
        pred[te] = ss[te] > c[int(np.argmax(bal))]
    trap = np.isin(cell[m], ["N_implicit", "N_explicit"])
    return dict(bal=float((pred[y == 1].mean() + (~pred[y == 0]).mean()) / 2), trap_wrong=float(pred[trap].mean()))
ch = choice[m]; y = truth[m]; trap = np.isin(cell[m], ["N_implicit", "N_explicit"])
out["rules"] = {"argmax": dict(bal=float((ch[y == 1].mean() + (~ch[y == 0]).mean()) / 2), trap_wrong=float(ch[trap].mean())),
                "tuned_head": rule(lp), "verify": rule(V)}
print(f"===== Jev {KIND} n={out['n']} cost=${out['cost']:.3f}")
print("  stop/proceed rate by cell:", {k: round(v, 2) for k, v in out["stop_or_proceed_rate"].items()})
for k, v in out["effects"].items(): print(f"  effect of {k}: {v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]")
print("  AUROC head/verify (all, matched):", {k: [round(x, 3) for x in v] for k, v in out["auroc"].items()})
print("  rules (balAcc, wrong on trap states):", {k: (round(v['bal'], 3), round(v['trap_wrong'], 3)) for k, v in out["rules"].items()})
json.dump(out, open(JL.replace(".jsonl", "_analysis.json"), "w"), indent=1)
