"""Analysis of the LLM-summary states (llm_summaries.py) scored by run_ac.py.

Cells: D_llm (done; LLM summary of the effective final action), N_llm_fail (not done; LLM summary of the same action
when the screen did not change), N_honest (not done; final action not logged). 'claim' on N_llm_fail = an LLM judge says
the failed-step summary asserts success.
Reports: stop rates; paired effects on log p(complete); head vs probe AUROC (D_llm vs N_llm_fail) on held-out apps,
including a probe trained only on the *templated* states (ac_decider4b.npz) of other apps -> transfer to LLM summaries.
usage: analyze_llm.py LLM_DATA LLM_NPZ TEMPLATED_DATA TEMPLATED_NPZ
"""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

rows = json.load(open(sys.argv[1])); z = np.load(sys.argv[2]); ops = list(z["ops"]); C = ops.index("complete")
P, H = z["P"], z["H"].astype(np.float32); lp = np.log(P[:, C] + 1e-6)
cell = np.array([r["cell"] for r in rows]); truth = np.array([r["truth"] for r in rows]); app = np.array([r["app"] for r in rows])
ep = np.array([r["ep"] for r in rows]); claim = np.array([r["claim"] for r in rows])
stop = P.argmax(1) == C; out = {}
fail = cell == "N_llm_fail"
out["llm_claims_success_on_failed_step"] = float(claim[fail].mean())
for c in ["D_llm", "N_llm_fail", "N_honest"]:
    out[f"stop_rate_{c}"] = float(stop[cell == c].mean())
out["stop_rate_N_llm_fail_claiming"] = float(stop[fail & (claim == 1)].mean()) if (fail & (claim == 1)).any() else None
out["stop_rate_N_llm_fail_honest"] = float(stop[fail & (claim == 0)].mean()) if (fail & (claim == 0)).any() else None

def paired(a, b):
    d = []
    for e in sorted(set(ep)):
        x, y = lp[(ep == e) & (cell == a)], lp[(ep == e) & (cell == b)]
        if len(x) and len(y): d.append(x[0] - y[0])
    d = np.array(d); bs = [np.random.default_rng(i).choice(d, len(d)).mean() for i in range(1000)]
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out["effect_logged_failed_step (N_llm_fail - N_honest)"] = paired("N_llm_fail", "N_honest")
out["effect_screen (D_llm - N_llm_fail)"] = paired("D_llm", "N_llm_fail")

m = np.isin(cell, ["D_llm", "N_llm_fail"]); gkf = GroupKFold(5)
def fit(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    return LogisticRegression(C=0.02, max_iter=3000).fit((Xtr - mu) / sd, ytr).predict_proba((Xte - mu) / sd)[:, 1]
out["head_auroc"] = float(roc_auc_score(truth[m], lp[m]))
best = (0, None)
for l in range(H.shape[1] // 3, H.shape[1]):
    pp = np.zeros(m.sum()); X, y, g = H[m][:, l], truth[m], app[m]
    for tr, te in gkf.split(X, y, g): pp[te] = fit(X[tr], y[tr], X[te])
    a = roc_auc_score(y, pp)
    if a > best[0]: best = (a, l)
out["probe_auroc_in_domain(best layer, optimistic)"] = [float(best[0]), int(best[1])]

# transfer: probe trained on templated states of OTHER apps (same layer index 10 = layer 20)
T_rows = json.load(open(sys.argv[3])); Tz = np.load(sys.argv[4]); TH = Tz["H"][:, 10].astype(np.float32)
Tapp = np.array([r["app"] for r in T_rows]); Ttruth = np.array([r["truth"] for r in T_rows])
pp = np.zeros(m.sum()); Xl, yl, gl = H[m][:, 10], truth[m], app[m]
for a in sorted(set(gl)):
    te = gl == a; tr = Tapp != a
    pp[te] = fit(TH[tr], Ttruth[tr], Xl[te])
out["probe_auroc_transfer_from_templated(layer20)"] = float(roc_auc_score(yl, pp))
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[2].replace(".npz", "_llm_analysis.json"), "w"), indent=1)
