"""Activation steering at the action slot, directions fit on half the goals and evaluated on the other half.

S1 claim-erasure : remove the 'history claims success' direction (diff of means) from the slot residual at layer l.
S2 truth-steer   : add alpha * (2*probe(h)-1) * u_truth at layer l (the model steers itself by its own latent belief).
S3 evidence-push : add -alpha * u_claim (constant push away from 'claimed'), to see if a plain CAA vector works.
"""
import json, sys, torch, numpy as np
from sklearn.linear_model import LogisticRegression
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle

MODEL, NPZ = sys.argv[1], sys.argv[2]
QA = "Which action should the agent take next to accomplish the goal?"
rows = json.load(open("results/data.json")); Hs = np.load(NPZ, allow_pickle=True)["H"].astype(np.float32)
truth = np.array([r["truth"] for r in rows]); claim = np.array([r["claim"] for r in rows]); gid = np.array([r["goal_id"] for r in rows])
gold = np.array([r["gold"] for r in rows]); si = np.array([r["stop_idx"] for r in rows])
m = DecisionModel(MODEL, grad_ckpt=False).cuda().eval(); tok = m.tok
items = [build(Example(r["ctx"], [Q(QA, r["options"], 0)]), tok, _NoShuffle(), max_options=255) for r in rows]
STATE = {}


def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    bi, sl = STATE["bi"], STATE["slot"]; x = h[bi, sl].float()
    kind, a = STATE["kind"], STATE["alpha"]
    if kind == "erase":
        u = STATE["uc"]; x = x - ((x @ u) - STATE["c0"])[:, None] * u          # project claim coordinate to its claim=0 mean
    elif kind == "truth":
        p = torch.sigmoid(x @ STATE["w"] + STATE["b"]); x = x + a * (2 * p - 1)[:, None] * STATE["ut"] * STATE["scale"]
    elif kind == "push":
        x = x - a * STATE["uc"] * STATE["scale"]
    h[bi, sl] = x.to(h.dtype)
    return out


@torch.no_grad()
def run(idx):
    P = []
    for s in range(0, len(idx), 16):
        its = [items[i] for i in idx[s:s + 16]]; T = max(len(it["ids"]) for it in its)
        ids = torch.full((len(its), T), tok.pad_token_id); att = torch.zeros_like(ids)
        for b, it in enumerate(its):
            ids[b, :len(it["ids"])] = torch.tensor(it["ids"]); att[b, :len(it["ids"])] = 1
        STATE["bi"] = torch.arange(len(its)).cuda(); STATE["slot"] = torch.tensor([it["slots"][0] for it in its]).cuda()
        h = m.lm.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state[STATE["bi"], STATE["slot"]]
        lg = m.cap(h.float() @ m.lm.lm_head.weight[m.letters].float().T)[:, :4] / 1.11
        P.append(torch.softmax(lg, -1).cpu().numpy())
    return np.concatenate(P)


def score(idx, P):
    pred = P.argmax(1); g = gold[idx]; t = truth[idx]; c = claim[idx]; s = si[idx]
    trap = ((pred == g) & (t == 0) & (c == 1)).sum() / ((t == 0) & (c == 1)).sum()
    fc = ((pred == s) & (t == 0)).sum() / (t == 0).sum(); mc = ((pred != s) & (t == 1)).sum() / (t == 1).sum()
    return (pred == g).mean(), trap, fc, mc


results = {}
halves = [gid < 12, gid >= 12]
for layer in [int(x) for x in sys.argv[3].split(",")]:
    mod = m.lm.model.layers[layer - 1]; hd = mod.register_forward_hook(hook)
    for cfg in [("none", 0)] + [("erase", 0)] + [("truth", a) for a in (2, 4, 8)] + [("push", a) for a in (1, 2, 4)]:
        agg = []
        for k in (0, 1):
            tr, te = np.where(halves[k])[0], np.where(halves[1 - k])[0]; X = Hs[tr, layer]
            mu1, mu0 = X[claim[tr] == 1].mean(0), X[claim[tr] == 0].mean(0); uc = (mu1 - mu0) / np.linalg.norm(mu1 - mu0)
            t1, t0 = X[truth[tr] == 1].mean(0), X[truth[tr] == 0].mean(0); ut = (t1 - t0) / np.linalg.norm(t1 - t0)
            mu, sd = X.mean(0), X.std(0) + 1e-4
            lr = LogisticRegression(C=0.05, max_iter=3000).fit((X - mu) / sd, truth[tr])
            w = lr.coef_[0] / sd; b = lr.intercept_[0] - (w * mu).sum()
            dev = lambda v: torch.tensor(v, dtype=torch.float32).cuda()
            STATE.update(kind=cfg[0], alpha=cfg[1], uc=dev(uc), ut=dev(ut), c0=float(X[claim[tr] == 0] @ uc).mean() if False else float((X[claim[tr] == 0] @ uc).mean()),
                         w=dev(w), b=float(b), scale=float(np.linalg.norm(t1 - t0)))
            agg.append((len(te), score(te, run(te))))
        n = sum(a for a, _ in agg); r = np.sum([a * np.array(s) for a, s in agg], 0) / n
        results[f"L{layer}-{cfg[0]}-{cfg[1]}"] = r.tolist()
        print(f"L{layer:2d} {cfg[0]:6s} a={cfg[1]:<3} acc={r[0]:.3f} trapAcc={r[1]:.3f} falseComp={r[2]:.3f} missedComp={r[3]:.3f}", flush=True)
    hd.remove()
json.dump(results, open(NPZ.replace(".npz", "_steer.json"), "w"), indent=1)
