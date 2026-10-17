"""Per-layer probe AUROC for Figure 2: decider-4b, Kev-4B and decider-12b in the three domains.
Phones: the saved per-layer probes (ac_*_analysis.json, 'probe'; held-out apps). ALFWorld/WebShop: computed here from the
hidden states with the same setup as analyze_alf.py (worked vs failed, F_honest excluded, GroupKFold(5) by game/goal).
Readout index -> layer: decider stores hidden_states[::2] (index k = layer 2k); Kev hooks every 2nd block (k = layer 2(k+1)).
Layers are reported as depth fractions so 32-layer and 48-layer models share an axis.
Output: results/layer_curves.json"""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
R = "results/"
MODELS = {"decider-4b": ("decider4b", "decider"), "Kev-4B": ("kev4b", "kev"), "decider-12b": ("decider12b", "decider")}
PHONE = {"decider-4b": "ac_decider4b", "Kev-4B": "ac_kev4b_full", "decider-12b": "ac_decider12b"}
WS_MAP = {"D_plain": "S_plain", "D_explicit": "S_think", "N_honest": "F_honest", "N_implicit": "F_implicit", "N_explicit": "F_explicit"}
def depth(k, n, kind): return (2 * k / (2 * (n - 1))) if kind == "decider" else (2 * (k + 1) / (2 * n))
out = {"phones": {}, "alfworld": {}, "webshop": {}}
for m, (tag, kind) in MODELS.items():
    a = json.load(open(R + PHONE[m] + "_analysis.json")); ks = sorted(int(k) for k in a["probe"]); n = len(ks)
    out["phones"][m] = dict(depth=[depth(k, n, kind) for k in ks], auroc=[a["probe"][str(k)][0] for k in ks],
                            head=a["rules"]["baseline p(complete) tuned threshold"]["auroc"])
for dom, data in [("alfworld", "alf_eval"), ("webshop", "ws_eval")]:
    rows = json.load(open(R + data + ".json"))
    cell = np.array([WS_MAP.get(r["cell"], r["cell"]) for r in rows]); keep = cell != "F_honest"
    y = np.array([r["truth"] for r in rows])[keep]; g = np.array([str(r.get("game", r.get("goal"))) for r in rows])[keep]
    for m, (tag, kind) in MODELS.items():
        H = np.load(f"{R}{data}_{tag}.npz")["H"].astype(np.float32)[keep]; n = H.shape[1]; curve = []
        for k in range(n):
            s = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(H, y, g):
                mu, sd = H[tr, k].mean(0), H[tr, k].std(0) + 1e-4
                s[te] = LogisticRegression(C=0.02, max_iter=3000).fit((H[tr, k] - mu) / sd, y[tr]).predict_proba((H[te, k] - mu) / sd)[:, 1]
            curve.append(float(roc_auc_score(y, s)))
        head = json.load(open(f"{R}{data}_{tag}_analysis.json"))["auroc"]["head proceed-vs-retry"]["all"]
        out[dom][m] = dict(depth=[depth(k, n, kind) for k in range(n)], auroc=curve, head=head)
        print(dom, m, "n_layers", n, "head %.3f" % head, " ".join("%.2f" % c for c in curve), flush=True)
for m in MODELS: print("phones", m, "head %.3f" % out["phones"][m]["head"], " ".join("%.2f" % c for c in out["phones"][m]["auroc"]))
json.dump(out, open(R + "layer_curves.json", "w"), indent=1)
