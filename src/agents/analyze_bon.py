"""Best-of-N curves from ws_bon.py. For N in 1..K, average over 300 random size-N subsets of each goal's K trajectories:
  oracle   : success if any trajectory in the subset succeeds (upper bound)
  random   : a random trajectory that bought (fail if none bought)
  decider  : the bought trajectory with the highest verifier p(yes)
  EviGate  : the bought trajectory with the highest probe score (probe on the verifier-slot layer-20 state, label = success,
             fit out-of-fold with GroupKFold by goal)
Also: share of picked trajectories with an UNEARNED claim (bought, the agent's thoughts claimed the options were selected,
but they were not).
usage: analyze_bon.py results/ws_bon.json
"""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

rows = json.load(open(sys.argv[1])); H = np.load(sys.argv[1].replace(".json", "_H.npy")).astype(np.float32)
goals = sorted({r["goal"] for r in rows}); K = max(r["k"] for r in rows) + 1
goals = [g for g in goals if sum(r["goal"] == g for r in rows) == K]            # complete goals only
rows = [r for r in rows if r["goal"] in goals]
b = [r for r in rows if r["bought"]]; hb = np.array([r["h_idx"] for r in b]); yb = np.array([r["success"] for r in b]).astype(int)
probe = {}
if len(set(yb)) > 1 and len(goals) >= 5:
    gb = np.array([r["goal"] for r in b]); s = np.zeros(len(b))
    for tr, te in GroupKFold(5).split(hb, yb, gb):
        X = H[hb]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-4
        if len(set(yb[tr])) < 2: s[te] = 0.5; continue
        s[te] = LogisticRegression(C=0.02, max_iter=3000).fit((X[tr] - mu) / sd, yb[tr]).predict_proba((X[te] - mu) / sd)[:, 1]
    probe = {(r["goal"], r["k"]): v for r, v in zip(b, s)}
unearned = lambda r: r["bought"] and r["claimed"] and not r["opts_ok"]
print(f"goals={len(goals)} K={K} trajectories={len(rows)} bought={len(b)} success={sum(r['success'] for r in rows)} "
      f"unearned-claim among bought={np.mean([unearned(r) for r in b]):.2f}")
if b:
    v = np.array([r["verifier_yes"] for r in b]); ue = np.array([unearned(r) for r in b])
    print(f"verifier p(yes): success {v[yb == 1].mean():.3f} | failed {v[yb == 0].mean():.3f} | failed WITH unearned claim {v[ue].mean() if ue.any() else float('nan'):.3f}")
rng = np.random.default_rng(0); out = {}
for N in [n for n in [1, 2, 4, 8, 16] if n <= K]:
    acc = {k: [] for k in ["oracle", "random", "decider", "EviGate"]}; ue_pick = {"decider": [], "EviGate": [], "random": []}
    for g in goals:
        G = [r for r in rows if r["goal"] == g]
        for _ in range(300):
            sub = [G[i] for i in rng.choice(K, N, replace=False)]; bs = [r for r in sub if r["bought"]]
            acc["oracle"].append(any(r["success"] for r in sub))
            if not bs:
                for k in ["random", "decider", "EviGate"]: acc[k].append(False)
                continue
            picks = {"random": bs[rng.integers(len(bs))], "decider": max(bs, key=lambda r: r["verifier_yes"])}
            if probe: picks["EviGate"] = max(bs, key=lambda r: probe[(r["goal"], r["k"])])
            for k, p in picks.items(): acc[k].append(p["success"]); ue_pick[k].append(unearned(p))
    out[N] = {k: float(np.mean(v)) for k, v in acc.items() if v}
    print(f"N={N:2d}  " + "  ".join(f"{k} {100 * v:5.1f}" for k, v in out[N].items()) +
          "   | picked has unearned claim: " + "  ".join(f"{k} {100 * np.mean(v):4.1f}" for k, v in ue_pick.items() if v))
json.dump(out, open(sys.argv[1].replace(".json", "_curves.json"), "w"), indent=1)
