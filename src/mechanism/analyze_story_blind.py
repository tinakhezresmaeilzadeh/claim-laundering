"""Analysis of story_blind.py outputs. Worked vs failed states (Honest cells excluded, as in the probe analysis).
Scores (higher = 'the step worked'):
  head      : log p(terminal) in the normal pass
  blind@l   : log p(terminal) when layers l.. cannot see the agent's self-report        (training-free)
  -reliance : -(head - blind@l), i.e. how little the decision rests on the self-report  (training-free)
  probe     : EviGate, linear probe on the layer-20 state, GroupKFold(5) by app/game/goal  (needs labels)
Gate rules on states where the model proposes the terminal option (argmax):
  flip@l    : veto iff the story-blind argmax is not the terminal option                  (training-free, no threshold)
  EviGate   : veto iff probe < balanced-accuracy threshold fit on training folds
usage: analyze_story_blind.py results/sb_{phones,alfworld,webshop}.json ...
"""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

out = {}
for f in sys.argv[1:]:
    rows = json.load(open(f)); H = np.load(f.replace(".json", ".npz"))["H"].astype(np.float32)
    keep = np.array([r["cell"] != "F_honest" for r in rows]); rows = [r for r, k in zip(rows, keep) if k]; H = H[keep]
    y = np.array([r["truth"] for r in rows]); g = np.array([r["app"] for r in rows]); cell = np.array([r["cell"] for r in rows])
    claimed = np.isin(cell, ["S_think", "F_explicit"]); LS = sorted(rows[0]["lp_blind"], key=int)
    T = lambda key, l=None: np.array([(r[key][r["term"]] if l is None else r[key][l][r["term"]]) for r in rows])
    head = T("lp_full"); blind = {l: T("lp_blind", l) for l in LS}
    am_full = np.array([int(np.argmax(r["lp_full"])) == r["term"] for r in rows])
    am_blind = {l: np.array([int(np.argmax(r["lp_blind"][l])) == r["term"] for r in rows]) for l in LS}
    probe = np.zeros(len(y)); gate = np.zeros(len(y), bool)
    for tr, te in GroupKFold(5).split(H, y, g):
        mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-4
        clf = LogisticRegression(C=0.02, max_iter=3000).fit((H[tr] - mu) / sd, y[tr])
        ptr = clf.predict_proba((H[tr] - mu) / sd)[:, 1]; probe[te] = clf.predict_proba((H[te] - mu) / sd)[:, 1]
        c = np.unique(np.quantile(ptr, np.linspace(0, 1, 201)))
        t = c[int(np.argmax([((ptr >= x)[y[tr] == 1].mean() + (ptr < x)[y[tr] == 0].mean()) / 2 for x in c]))]
        gate[te] = probe[te] < t
    auc = lambda s, k=None: float(roc_auc_score(y if k is None else y[k], s if k is None else s[k]))
    dom = f.rsplit("/", 1)[-1].split("_", 1)[1].split(".")[0]; o = {"n": int(len(y)), "dropped_tokens_mean": float(np.mean([r["n_drop"] for r in rows]))}
    o["auroc"] = {"head": [auc(head), auc(head, claimed)], "probe (EviGate)": [auc(probe), auc(probe, claimed)]}
    for l in LS:
        o["auroc"][f"blind@{l}"] = [auc(blind[l]), auc(blind[l], claimed)]
        o["auroc"][f"-reliance@{l}"] = [auc(blind[l] - head), auc(blind[l] - head, claimed)]
    # gate behaviour on terminal proposals: wrong terminal decisions left on failed states, vetoes of worked states
    term_f, term_w = am_full & (y == 0), am_full & (y == 1)
    def rule(veto):
        return dict(failed_pass=float((term_f & ~veto).sum() / max((y == 0).sum(), 1)),
                    worked_vetoed=float((term_w & veto).sum() / max(term_w.sum(), 1)),
                    claimed_failed_pass=float((term_f & ~veto & claimed).sum() / max(((y == 0) & claimed).sum(), 1)))
    o["gate"] = {"none": rule(np.zeros(len(y), bool)), "EviGate": rule(gate)}
    for l in LS: o["gate"][f"flip@{l}"] = rule(~am_blind[l])
    out[dom] = o
    print(f"\n===== {dom}  (n={len(y)} worked/failed states, mean {o['dropped_tokens_mean']:.0f} self-report tokens dropped)")
    print("AUROC  [all, claimed-only]")
    for k, v in o["auroc"].items(): print(f"  {k:16s} {v[0]:.3f}  {v[1]:.3f}")
    print("Gate on terminal proposals: failed steps that still pass / worked steps vetoed / claimed failed that pass")
    for k, v in o["gate"].items(): print(f"  {k:10s} {v['failed_pass']:.3f}  {v['worked_vetoed']:.3f}  {v['claimed_failed_pass']:.3f}")
json.dump(out, open(sys.argv[1].rsplit("/", 1)[0] + "/" + sys.argv[1].rsplit("/", 1)[-1].split("_", 1)[0] + "_analysis.json", "w"), indent=1)
