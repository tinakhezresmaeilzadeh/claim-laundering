"""ALFWorld recovery analysis (build_alf.py + run_alf.py).

Decision: proceed to the next plan step (a_{k+1}) vs retry the last action (a_k). Truth = the last action worked.
Effects on log p(proceed), paired within (game, step):
  evidence       S_plain - F_implicit   (same history; observation real vs 'Nothing happens.')
  implicit claim F_implicit - F_honest  (same state; the failed attempt is logged)
  explicit claim F_explicit - F_implicit (same observation; a think: line claims success)
Discrimination of worked vs failed (S_* vs F_implicit/F_explicit): head = p(proceed)/(p(proceed)+p(retry)), verify question,
nested probe (GroupKFold by game, layer chosen on training games), TF-IDF on the transcript tail; matched = S_think vs F_explicit.
Stop-rule analogue: proceed-or-retry decision with thresholds fit on training folds.
usage: analyze_alf.py DATA NPZ NAME
"""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer

rows = json.load(open(sys.argv[1])); z = np.load(sys.argv[2]); name = sys.argv[3]
WS_MAP = {"D_plain": "S_plain", "D_explicit": "S_think", "N_honest": "F_honest", "N_implicit": "F_implicit", "N_explicit": "F_explicit"}
for r in rows:                                   # WebShop rows (build_ws.py): same five-cell design, 'buy now' = proceed
    if r["cell"] in WS_MAP: r["cell"] = WS_MAP[r["cell"]]; r.setdefault("game", r["goal"]); r.setdefault("step", 0)
P, V, H = z["P"], z["V"], z["H"].astype(np.float32)
cell = np.array([r["cell"] for r in rows]); truth = np.array([r["truth"] for r in rows]); grp = np.array([r["game"] for r in rows])
key = np.array([f"{r['game']}_{r['step']}" for r in rows])
pp_ = np.array([p[r["proceed"]] for p, r in zip(P, rows)]); pr_ = np.array([p[r["retry"]] for p, r in zip(P, rows)])
lpro = np.log(pp_ + 1e-6); head = np.log(pp_ + 1e-9) - np.log(pr_ + 1e-9)
out = {"model": name, "n": len(rows), "games": len(set(grp))}
print(f"===== {name} proceed-vs-fix (n={len(rows)}, {len(set(grp))} games)")
CELLS = ["S_plain", "S_think", "F_honest", "F_implicit", "F_explicit"]
prop = P.argmax(1) == np.array([r["proceed"] for r in rows])
out["proceed_rate"] = {c: float(prop[cell == c].mean()) for c in CELLS}
print("argmax=proceed by cell:", {c: round(v, 2) for c, v in out["proceed_rate"].items()})

def paired(a, b, x):
    d = np.array([x[(key == k) & (cell == a)][0] - x[(key == k) & (cell == b)][0] for k in sorted(set(key))])
    bs = [np.random.default_rng(i).choice(d, len(d)).mean() for i in range(1000)]
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out["effects"] = {"evidence": paired("S_plain", "F_implicit", lpro), "implicit_claim": paired("F_implicit", "F_honest", lpro),
                  "explicit_claim": paired("F_explicit", "F_implicit", lpro)}
for k, v in out["effects"].items(): print(f"  effect on log p(proceed) of {k}: {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]")

m = cell != "F_honest"; y = truth[m]; g = grp[m]; EXM = np.isin(cell[m], ["S_think", "F_explicit"]); gkf = GroupKFold(5)
def fit(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    return LogisticRegression(C=0.02, max_iter=3000).fit((Xtr - mu) / sd, ytr).predict_proba((Xte - mu) / sd)[:, 1]
Hm = H[m]; nested = np.zeros(m.sum()); L = Hm.shape[1]
for tr, te in gkf.split(Hm, y, g):
    best = (-1, None)
    for l in range(L // 3, L):
        p = np.zeros(len(tr))
        for itr, ite in GroupKFold(4).split(tr, y[tr], g[tr]): p[ite] = fit(Hm[tr][itr, l], y[tr][itr], Hm[tr][ite, l])
        a = roc_auc_score(y[tr], p)
        if a > best[0]: best = (a, l)
    nested[te] = fit(Hm[tr, best[1]], y[tr], Hm[te, best[1]])
texts = [r["prompt"][-1500:] for r, k in zip(rows, m) if k]; tf = np.zeros(m.sum())
for tr, te in gkf.split(texts, y, g):
    v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True); Xtr = v.fit_transform([texts[i] for i in tr])
    tf[te] = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, y[tr]).predict_proba(v.transform([texts[i] for i in te]))[:, 1]
auc = lambda s, k=None: float(roc_auc_score(y if k is None else y[k], s if k is None else s[k]))
out["auroc"] = {n_: dict(all=auc(s), matched=auc(s, EXM)) for n_, s in
                [("head proceed-vs-retry", head[m]), ("verify question", V[m]), ("nested probe", nested), ("TF-IDF transcript tail", tf)]}
for k, v in out["auroc"].items(): print(f"  AUROC {k:24s} all={v['all']:.3f} matched={v['matched']:.3f}")
gs = np.array(sorted(set(g))); idx = {a: np.where(g == a)[0] for a in gs}; rng = np.random.default_rng(0); d1, d2 = [], []
for _ in range(1000):
    b = np.concatenate([idx[a] for a in rng.choice(gs, len(gs))]); t = y[b]
    if len(set(t)) < 2: continue
    d1.append(roc_auc_score(t, nested[b]) - roc_auc_score(t, head[m][b])); e = EXM[b]
    if len(set(t[e])) > 1: d2.append(roc_auc_score(t[e], nested[b][e]) - roc_auc_score(t[e], head[m][b][e]))
out["gap"] = {"all": [float(np.mean(d1)), float(np.percentile(d1, 2.5)), float(np.percentile(d1, 97.5))],
              "matched": [float(np.mean(d2)), float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))]}
print("  probe - head gap:", {k: [round(x, 3) for x in v] for k, v in out["gap"].items()})
# recovery decision on held-out games: proceed iff score > threshold fit on train folds
def rule(s):
    pred = np.zeros(len(s), bool)
    for tr, te in gkf.split(s, y, g):
        c = np.unique(np.quantile(s[tr], np.linspace(0, 1, 201)))
        bal = [((s[tr] > t)[y[tr] == 1].mean() + (s[tr] <= t)[y[tr] == 0].mean()) / 2 for t in c]
        pred[te] = s[te] > c[int(np.argmax(bal))]
    return float(((pred[y == 1]).mean() + (~pred[y == 0]).mean()) / 2), float(pred[y == 0].mean())
out["recovery_rule"] = {"argmax": (float((prop[m][y == 1].mean() + (~prop[m][y == 0]).mean()) / 2), float(prop[m][y == 0].mean())),
                        "tuned head": rule(head[m]), "verify": rule(V[m]), "EviGate": rule(nested)}
print("  recovery (balAcc, failed-but-proceeds rate):", {k: tuple(round(x, 3) for x in v) for k, v in out["recovery_rule"].items()})
json.dump(out, open(sys.argv[2].replace(".npz", "_analysis.json"), "w"), indent=1)
