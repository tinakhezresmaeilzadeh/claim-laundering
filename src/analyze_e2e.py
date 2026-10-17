"""Pool the two e2e folds (24 goals x SEEDS), paired comparisons against no gate on identical (goal, seed) episodes.
CI: bootstrap over goals (task-cluster), as in arXiv 2610.13092."""
import json, sys, numpy as np
S = sys.argv[1] if len(sys.argv) > 1 else "30"
F = [json.load(open(f"results/e2e_sim_s{S}_f{k}.json")) for k in (0, 1)]
gates = ["none", "verify", "contrast", "probe", "random"]
ep = {g: F[0][g]["episodes"] + F[1][g]["episodes"] for g in gates}
key = lambda e: (e["goal"], e["seed"])
goals = sorted({e["goal"] for e in ep["none"]})
def per_goal(g, f):
    return np.array([np.mean([f(e) for e in ep[g] if e["goal"] == q]) for q in goals])
rng = np.random.default_rng(0); B = [rng.integers(0, len(goals), len(goals)) for _ in range(5000)]
out = {}
base = per_goal("none", lambda e: e["success"])
print(f"{'gate':9s} success [95% CI]        Δ vs none [95% CI]      falseStop  timeout  calls/ep  calls/success  stop-reject")
for g in gates:
    s = per_goal(g, lambda e: e["success"]); d = s - base
    ci = np.percentile([s[b].mean() for b in B], [2.5, 97.5]); dci = np.percentile([d[b].mean() for b in B], [2.5, 97.5])
    fs = np.mean([e["false_stop"] for e in ep[g]]); to = np.mean([not e["stopped"] for e in ep[g]])
    calls = np.mean([e["calls"] for e in ep[g]]); cps = calls * len(ep[g]) / max(sum(e["success"] for e in ep[g]), 1)
    rj = sum(e["rej"] for e in ep[g]) / max(sum(e["props"] for e in ep[g]), 1)
    p_one = np.mean([d[b].mean() <= 0 for b in B]) if g != "none" else np.nan
    out[g] = dict(success=s.mean(), ci=list(ci), delta=d.mean(), delta_ci=list(dci), p_delta_le_0=p_one, false_stop=fs, timeout=to,
                  calls_per_episode=calls, calls_per_success=cps, stop_reject_rate=rj)
    print(f"{g:9s} {s.mean():.3f} [{ci[0]:.3f},{ci[1]:.3f}]   {d.mean():+.3f} [{dci[0]:+.3f},{dci[1]:+.3f}] p={p_one:.3f}   "
          f"{fs:.3f}     {to:.3f}    {calls:.2f}      {cps:.2f}          {rj:.3f}")
json.dump(out, open(f"results/e2e_pooled_s{S}.json", "w"), indent=1, default=float)
