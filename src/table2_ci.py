"""Uncertainty for Table 2: paired bootstrap CIs (resampling decision points) for, in every model x domain configuration,
  obs, logged, claim          : the three effects on log p(terminal option), as reported in Table 2 (reproduced as a check)
  logged - obs, claim - obs   : does each record component move the decision more than the observation?
  obs | claim                 : the observation's effect when the false claim is present (Done+claim - Logged+claim)
Cells: D_plain (Done), D_explicit (Done+claim), N_honest, N_implicit (Logged), N_explicit (Logged+claim).
Open models: saved probabilities (P) of the terminal option; hosted Jev: the API's probabilities (same clipping as analyze_jev).
Output: results/table2_ci.json
"""
import json, numpy as np
R = "results/"
MAP = {"S_plain": "D_plain", "S_think": "D_explicit", "F_honest": "N_honest", "F_implicit": "N_implicit", "F_explicit": "N_explicit"}


def open_model(data, npz, dom):
    rows = json.load(open(R + data)); P = np.load(R + npz)["P"]
    if dom == "phones":
        ops = list(np.load(R + npz)["ops"]); C = ops.index("complete")
        lp = np.log(P[:, C] + 1e-6); key = [str(r["ep"]) for r in rows]
    else:
        lp = np.log(np.array([p[r["proceed"]] for p, r in zip(P, rows)]) + 1e-6)
        key = [f"{r.get('game', r.get('goal'))}_{r.get('step', 0)}" for r in rows]
    return lp, key, [MAP.get(r["cell"], r["cell"]) for r in rows]


def jev(kind, data):
    rows = json.load(open(R + data)); recs = {r["i"]: r for r in map(json.loads, open(R + f"jev_{kind}.jsonl"))}
    idx = [i for i in range(len(rows)) if i in recs and recs[i]["op"] and recs[i]["ver"]]
    rows = [rows[i] for i in idx]; X = [recs[i] for i in idx]
    pk = (lambda r: "complete") if kind == "ac" else (lambda r: f"A{r['proceed']}")
    lp = np.log(np.clip(np.array([x["op"]["probabilities"].get(pk(r), 0.0) for r, x in zip(rows, X)]), 1e-4, 1))
    key = [str(r["ep"]) if kind == "ac" else f"{r.get('game', r.get('goal'))}_{r.get('step', 0)}" for r in rows]
    return lp, key, [MAP.get(r["cell"], r["cell"]) for r in rows]


CONF = [("phones", "decider-4b", lambda: open_model("ac.json", "ac_decider4b.npz", "phones")),
        ("phones", "Kev-4B", lambda: open_model("ac.json", "ac_kev4b_full.npz", "phones")),
        ("phones", "decider-12b", lambda: open_model("ac_sub.json", "ac_decider12b.npz", "phones")),
        ("phones", "hosted Jev", lambda: jev("ac", "ac_sub.json")),
        ("alfworld", "decider-4b", lambda: open_model("alf_eval.json", "alf_eval_decider4b.npz", "alf")),
        ("alfworld", "Kev-4B", lambda: open_model("alf_eval.json", "alf_eval_kev4b.npz", "alf")),
        ("alfworld", "decider-12b", lambda: open_model("alf_eval.json", "alf_eval_decider12b.npz", "alf")),
        ("alfworld", "hosted Jev", lambda: jev("alf", "alf_eval.json")),
        ("webshop", "decider-4b", lambda: open_model("ws_eval.json", "ws_eval_decider4b.npz", "ws")),
        ("webshop", "Kev-4B", lambda: open_model("ws_eval.json", "ws_eval_kev4b.npz", "ws")),
        ("webshop", "decider-12b", lambda: open_model("ws_eval.json", "ws_eval_decider12b.npz", "ws")),
        ("webshop", "hosted Jev", lambda: jev("ws", "ws_eval.json"))]
CELLS = ["D_plain", "D_explicit", "N_honest", "N_implicit", "N_explicit"]
out = {}
for dom, model, load in CONF:
    lp, key, cell = load(); by = {}
    for v, k, c in zip(lp, key, cell): by.setdefault(k, {})[c] = v
    K = [k for k in by if all(c in by[k] for c in CELLS)]
    A = {c: np.array([by[k][c] for k in K]) for c in CELLS}
    d = {"obs": A["D_plain"] - A["N_implicit"], "logged": A["N_implicit"] - A["N_honest"], "claim": A["N_explicit"] - A["N_implicit"],
         "obs_given_claim": A["D_explicit"] - A["N_explicit"]}
    d["logged_minus_obs"] = d["logged"] - d["obs"]; d["claim_minus_obs"] = d["claim"] - d["obs"]
    rng = np.random.default_rng(0); B = rng.integers(0, len(K), (2000, len(K)))
    res = {k: [float(v.mean()), float(np.percentile(v[B].mean(1), 2.5)), float(np.percentile(v[B].mean(1), 97.5))] for k, v in d.items()}
    res["n_points"] = len(K); out[f"{dom}/{model}"] = res
    best = "logged_minus_obs" if res["logged"][0] >= res["claim"][0] else "claim_minus_obs"
    sig = res[best][1] > 0
    print(f"{dom:8s} {model:12s} n={len(K):4d} | obs {res['obs'][0]:+.2f} logged {res['logged'][0]:+.2f} claim {res['claim'][0]:+.2f} | "
          f"logged-obs {res['logged_minus_obs'][0]:+.2f} [{res['logged_minus_obs'][1]:+.2f},{res['logged_minus_obs'][2]:+.2f}] "
          f"claim-obs {res['claim_minus_obs'][0]:+.2f} [{res['claim_minus_obs'][1]:+.2f},{res['claim_minus_obs'][2]:+.2f}] "
          f"| obs|claim {res['obs_given_claim'][0]:+.2f} [{res['obs_given_claim'][1]:+.2f},{res['obs_given_claim'][2]:+.2f}] | larger record effect > obs: {'YES' if sig else 'not significant'}")
json.dump(out, open(R + "table2_ci.json", "w"), indent=1)
