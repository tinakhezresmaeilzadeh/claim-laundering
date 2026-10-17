"""Merge the interrupted live-calibrated WebShop run (goals 0-134, from its log) with the resumed part (goals 135-199)."""
import json, re, numpy as np
eps = []
for l in open("results/ws_e2e_probe2_part1.log"):
    m = re.match(r"^(\d+) reward=([\d.]+) bought=(\w+) steps=(\d+) inj=(\d+) trig=(\d+) veto=(\d+)", l)
    if m:
        g, r, b, s, i, t, v = m.groups()
        eps.append(dict(goal=int(g), reward=float(r), success=float(r) >= 1.0, bought=b == "True", steps=int(s), injected=int(i), triggers=int(t), vetoes=int(v)))
eps = [e for e in eps if e["goal"] < 135] + json.load(open("results/ws_e2e_probe2b.json"))["episodes"]
eps = sorted({e["goal"]: e for e in eps}.values(), key=lambda e: e["goal"])
summ = dict(gate="probe2", n=len(eps), success=float(np.mean([e["success"] for e in eps])), score=float(np.mean([e["reward"] for e in eps])),
            veto_rate=sum(e["vetoes"] for e in eps) / max(sum(e["triggers"] for e in eps), 1))
json.dump(dict(summary=summ, episodes=eps), open("results/ws_e2e_probe2.json", "w"), indent=1); print(summ)
