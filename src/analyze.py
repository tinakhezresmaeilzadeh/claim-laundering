"""Claim-vs-evidence diagnostics, latent probes, and training-free fixes."""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

T = 1.11
rows = json.load(open("results/data.json"))
z = np.load(sys.argv[1], allow_pickle=True)
A, A0, V, H = [np.asarray(x, dtype=np.float64) for x in (z["A"], z["A0"], z["V"])] + [z["H"]]
truth = np.array([r["truth"] for r in rows]); claim = np.array([r["claim"] for r in rows]); gid = np.array([r["goal_id"] for r in rows])
gold = np.array([r["gold"] for r in rows]); si = np.array([r["stop_idx"] for r in rows])
sm = lambda x: (lambda e: e / e.sum())(np.exp((x - x.max()) / T))
PA = [sm(a) for a in A]; PA0 = [sm(a) for a in A0]
pstop = np.array([p[s] for p, s in zip(PA, si)]); pver = np.array([sm(v)[0] for v in V])


def report(name, P):
    pred = np.array([p.argmax() for p in P]); ps = np.array([p[s] for p, s in zip(P, si)])
    acc = (pred == gold).mean()
    fc = ((pred == si) & (truth == 0)).mean() / (truth == 0).mean()          # false completion among not-done
    mc = ((pred != si) & (truth == 1)).mean() / (truth == 1).mean()          # missed completion among done
    cells = " ".join(f"T{t}C{c}:{(pred == gold)[(truth == t) & (claim == c)].mean():.2f}" for t in (0, 1) for c in (0, 1))
    print(f"{name:34s} acc={acc:.3f} falseComp={fc:.3f} missedComp={mc:.3f} AUROC(stop)={roc_auc_score(truth, ps):.3f} | {cells}")


print("== Behaviour (cell accuracy: T=truth done, C=history claims success) ==")
report("head (baseline)", PA)
print(f"mean p(stop) by cell: " + " ".join(f"T{t}C{c}={pstop[(truth == t) & (claim == c)].mean():.3f}" for t in (0, 1) for c in (0, 1)))
eff_claim = pstop[claim == 1].mean() - pstop[claim == 0].mean(); eff_ev = pstop[truth == 1].mean() - pstop[truth == 0].mean()
print(f"effect of claim on p(stop)={eff_claim:+.3f}   effect of screen evidence={eff_ev:+.3f}")
print(f"verify-question: acc={((pver > .5) == truth).mean():.3f} AUROC={roc_auc_score(truth, pver):.3f} | " +
      " ".join(f"T{t}C{c}:{((pver > .5) == truth)[(truth == t) & (claim == c)].mean():.2f}" for t in (0, 1) for c in (0, 1)))

print("\n== Latent probe on action-slot hidden state (GroupKFold by goal, held-out goals) ==")
L = H.shape[1]; gkf = GroupKFold(6); best = (0, 0)
probe_p = {}
for l in range(0, L, 2):
    X = H[:, l].astype(np.float32); X = (X - X.mean(0)) / (X.std(0) + 1e-4); pp = np.zeros(len(X)); pc = np.zeros(len(X))
    for tr, te in gkf.split(X, truth, gid):
        pp[te] = LogisticRegression(C=0.05, max_iter=2000).fit(X[tr], truth[tr]).predict_proba(X[te])[:, 1]
        pc[te] = LogisticRegression(C=0.05, max_iter=2000).fit(X[tr], claim[tr]).predict_proba(X[te])[:, 1]
    a = roc_auc_score(truth, pp); probe_p[l] = pp
    print(f"layer {l:2d}: probe AUROC truth={a:.3f}  claim={roc_auc_score(claim, pc):.3f}  truth-acc in claim&notdone={((pp > .5) == truth)[(claim == 1) & (truth == 0)].mean():.2f}")
    if a > best[0]: best = (a, l)
print("best layer", best)

print("\n== Training-free fixes ==")
def couple(pv, beta=1.0):                     # replace stop mass by the verify slot's p(done); keep action ranking
    out = []
    for p, s, v in zip(PA, si, pv):
        q = p.copy(); rest = 1 - p[s]; w = v ** beta
        q = q * (1 - w) / max(rest, 1e-9); q[s] = w; out.append(q)
    return out
report("F1 couple stop<-verify slot", couple(pver))
for al in (0.5, 1.0, 2.0):
    report(f"F2 evidence-contrast a={al}", [sm(a + al * (a - a0)) for a, a0 in zip(A, A0)])
report(f"F3 couple stop<-probe L{best[1]}", couple(probe_p[best[1]]))
# gated: only intervene when the model is about to stop (selective verification)
pv_g = [v if p.argmax() == s else p[s] for p, s, v in zip(PA, si, pver)]
report("F1-gated (only when head says stop)", couple(np.array(pv_g)))
