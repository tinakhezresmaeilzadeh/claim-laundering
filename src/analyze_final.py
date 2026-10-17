"""Finalization checks for one model's AndroidControl run (all evaluation on held-out apps).

1. Nested probe: layer chosen by inner GroupKFold on the training apps only (no test-set layer picking).
2. Paired app-level bootstrap CI for AUROC(probe) - AUROC(output head), overall and on the explicit-claim-matched pair.
3. Text-only controls: TF-IDF logistic regression on (a) the screen text only, (b) the full prompt. If (a) matches the
   probe, the probe's advantage is reading surface screen features any classifier could read.
4. Probe learning curve over the number of training episodes.
5. (decider-4b only) synthetic->real transfer: probe fit on the synthetic set, applied to real states without real labels.
usage: analyze_final.py DATA NPZ NAME [SYNTH_NPZ SYNTH_LAYER REAL_IDX]
"""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer

rows = json.load(open(sys.argv[1])); z = np.load(sys.argv[2]); name = sys.argv[3]
ops = list(z["ops"]); C = ops.index("complete"); P, V, H = z["P"], z["V"], z["H"].astype(np.float32)
truth = np.array([r["truth"] for r in rows]); app = np.array([r["app"] for r in rows]); cell = np.array([r["cell"] for r in rows])
ep = np.array([r["ep"] for r in rows]); head = np.log(P[:, C] + 1e-6)
EXM = np.isin(cell, ["D_explicit", "N_explicit"])
out = {"model": name}
gkf = GroupKFold(5)


def fit_pred(Xtr, ytr, Xte, Cr=0.02):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
    return LogisticRegression(C=Cr, max_iter=3000).fit((Xtr - mu) / sd, ytr).predict_proba((Xte - mu) / sd)[:, 1]


# 1. nested probe
nested = np.zeros(len(rows)); chosen = []
L = H.shape[1]; cand = list(range(L // 3, L))                # skip the earliest third (never informative in the pilots)
for tr, te in gkf.split(H, truth, app):
    inner = GroupKFold(4); best = (-1, None)
    for l in cand:
        pp = np.zeros(len(tr))
        for itr, ite in inner.split(tr, truth[tr], app[tr]):
            pp[ite] = fit_pred(H[tr][itr, l], truth[tr][itr], H[tr][ite, l])
        a = roc_auc_score(truth[tr], pp)
        if a > best[0]: best = (a, l)
    chosen.append(best[1]); nested[te] = fit_pred(H[tr, best[1]], truth[tr], H[te, best[1]])
out["nested_layers"] = chosen

# 3. text-only controls
def screen(p):
    s = p.split("current screen:\n", 1)[1]; return s.split("\nNote that", 1)[0][:20000]
def tfidf(texts):
    pr = np.zeros(len(rows))
    for tr, te in gkf.split(texts, truth, app):
        v = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
        Xtr = v.fit_transform([texts[i] for i in tr]); Xte = v.transform([texts[i] for i in te])
        pr[te] = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, truth[tr]).predict_proba(Xte)[:, 1]
    return pr
tf_screen = tfidf([screen(r["prompt"]) for r in rows]); tf_full = tfidf([r["prompt"] for r in rows])

# 2. AUROCs + paired app bootstrap
apps = np.array(sorted(set(app))); idx = {a: np.where(app == a)[0] for a in apps}
def auc(s, m=None):
    m = np.ones(len(s), bool) if m is None else m
    return roc_auc_score(truth[m], s[m]) if len(set(truth[m])) > 1 else np.nan
scores = {"output head p(complete)": head, "coupled verify question": V, "nested probe": nested,
          "TF-IDF screen-only": tf_screen, "TF-IDF full prompt": tf_full}
out["auroc"] = {k: dict(all=auc(s), explicit_matched=auc(s, EXM)) for k, s in scores.items()}
rng = np.random.default_rng(0); diffs = {"all": [], "explicit_matched": []}; diffs_tf = []
for _ in range(1000):
    b = np.concatenate([idx[a] for a in rng.choice(apps, len(apps))])
    m = np.zeros(len(rows), bool); w = np.bincount(b, minlength=len(rows))
    sel = np.repeat(np.arange(len(rows)), w)
    t = truth[sel]
    if len(set(t)) < 2: continue
    diffs["all"].append(roc_auc_score(t, nested[sel]) - roc_auc_score(t, head[sel]))
    diffs_tf.append(roc_auc_score(t, nested[sel]) - roc_auc_score(t, tf_screen[sel]))
    e = EXM[sel]
    if len(set(t[e])) > 1: diffs["explicit_matched"].append(roc_auc_score(t[e], nested[sel][e]) - roc_auc_score(t[e], head[sel][e]))
out["gap_probe_minus_head"] = {k: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in diffs.items()}
out["gap_probe_minus_tfidf_screen"] = [float(np.mean(diffs_tf)), float(np.percentile(diffs_tf, 2.5)), float(np.percentile(diffs_tf, 97.5))]

# 4. learning curve (layer = most frequently chosen nested layer)
lc_layer = max(set(chosen), key=chosen.count); out["learning_curve"] = {}
for n_ep in (10, 25, 50, 100, 10 ** 9):
    pr = np.zeros(len(rows)); ok = True
    for k, (tr, te) in enumerate(gkf.split(H, truth, app)):
        eps_tr = np.array(sorted(set(ep[tr]))); r = np.random.default_rng(k)
        keep = set(r.choice(eps_tr, min(n_ep, len(eps_tr)), replace=False)); tr2 = tr[np.isin(ep[tr], list(keep))]
        pr[te] = fit_pred(H[tr2, lc_layer], truth[tr2], H[te, lc_layer])
    out["learning_curve"][min(n_ep, len(set(ep)))] = auc(pr)

# 5. synthetic -> real transfer
if len(sys.argv) > 4:
    S = np.load(sys.argv[4], allow_pickle=True)["H"][:, int(sys.argv[5])].astype(np.float32)
    ys = np.array([r["truth"] for r in json.load(open("results/data.json"))])
    out["synthetic_to_real_auroc"] = dict(all=auc(fit_pred(S, ys, H[:, int(sys.argv[6])])), explicit_matched=auc(fit_pred(S, ys, H[:, int(sys.argv[6])]), EXM))

print(f"===== {name}")
print("nested layers chosen per fold:", chosen)
for k, v in out["auroc"].items(): print(f"  AUROC {k:26s} all={v['all']:.3f}  explicit-matched={v['explicit_matched']:.3f}")
for k, v in out["gap_probe_minus_head"].items(): print(f"  probe - head gap ({k}): {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
v = out["gap_probe_minus_tfidf_screen"]; print(f"  probe - TF-IDF(screen) gap: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
print("  learning curve (train episodes -> AUROC):", {k: round(v, 3) for k, v in out["learning_curve"].items()})
if "synthetic_to_real_auroc" in out: print("  synthetic->real probe transfer:", {k: round(v, 3) for k, v in out["synthetic_to_real_auroc"].items()})
json.dump(out, open(sys.argv[2].replace(".npz", "_final.json"), "w"), indent=1, default=float)
