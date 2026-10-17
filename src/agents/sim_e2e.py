"""Closed-loop end-to-end test: a text phone simulator with flaky actions and an optimistic step summarizer.

The agent (decider-4b) picks one action per step among candidate actions plus 'Stop: the task is complete'. The tap that
achieves the goal fails silently with probability P_FAIL (screen unchanged); the summarizer still writes a success claim
with probability P_CLAIM_ON_FAIL (LLM summaries often assume the intended effect). Success = goal state reached when the
agent stops (or at the step limit, the run is a failure).

Stop gates (applied only when the model's argmax is Stop):
  none      accept the stop
  verify    coupled verify question in the same request; accept if p(yes) >= 0.5, else take the best non-stop action
  contrast  evidence-contrast scores (alpha=2, screen-ablated pass) for the whole decision
  probe     linear probe on the layer-20 hidden state at the action slot (fit on synthetic data, goals 0-11 only)
  random    reject the stop with probability equal to verify's measured rejection rate (budget-matched control)
Evaluated on goals 12-23 (held out from the probe).
"""
import json, random, sys, numpy as np, torch
from sklearn.linear_model import LogisticRegression
sys.path[:0] = ["src/common", "src/data"]
from gen_data import G, NOISE, STOP
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle

P_FAIL, P_CLAIM_ON_FAIL, MAX_STEPS, SEEDS = 0.4, 0.7, 8, int(sys.argv[1]) if len(sys.argv) > 1 else 20
FOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 0      # 0: probe fit on goals 0-11, eval 12-23; 1: the reverse
QA = "Which action should the agent take next to accomplish the goal?"
QV = "Does the current screen show that the goal has already been accomplished?"
APPS = sorted({g[1].split(" > ")[0] for g in G})
m = DecisionModel("Mapika/decider-4b", grad_ckpt=False).cuda().eval(); tok = m.tok; T = 1.11

# probe fit on synthetic states of goals 0-11 (layer 20 of the saved hidden states)
rows = json.load(open("results/data.json")); Hs = np.load("results/decider4b.npz", allow_pickle=True)["H"][:, 20].astype(np.float32)
tr = np.array([(r["goal_id"] < 12) == (FOLD == 0) for r in rows]); y = np.array([r["truth"] for r in rows])
mu, sd = Hs[tr].mean(0), Hs[tr].std(0) + 1e-4
probe = LogisticRegression(C=0.05, max_iter=3000).fit((Hs[tr] - mu) / sd, y[tr])


@torch.no_grad()
def ask(ctx, q, opts, want_h=False):
    it = build(Example(ctx, [Q(q, opts, 0)]), tok, _NoShuffle(), max_options=255)
    out = m.lm.model(input_ids=torch.tensor([it["ids"]]).cuda(), output_hidden_states=want_h)
    h = out.last_hidden_state[0, it["slots"][0]]
    lg = m.cap(h.float() @ m.lm.lm_head.weight[m.letters].float().T)[: it["nopts"][0]].cpu().numpy() / T
    H = out.hidden_states[20][0, it["slots"][0]].float().cpu().numpy() if want_h else None
    return lg, H


def sm(x): e = np.exp(x - x.max()); return e / e.sum()


def episode(g, gate, rng, reject_rate):
    goal, app_path, s0, s1, el, act, dis = g; app = app_path.split(" > ")[0]
    where, done, hist, calls = "home", False, [], 0
    noise = rng.sample(NOISE, 4); tapped = 0; props = rej = 0
    for step in range(MAX_STEPS):
        if where == "home":
            screen = [f"App icon: {a}" for a in APPS] + noise[:2]
            cands = [f"Open the {a} app" for a in APPS] + [STOP]
        else:
            screen = [s1 if done else s0, f"Button: {el}"] + [f"Button: {d.split(chr(39))[1] if chr(39) in d else d}" for d in dis] + noise
            cands = [act] + dis + ["Navigate back", STOP]
        rng.shuffle(screen); order = cands[:]; rng.shuffle(order)
        hdr = f"Goal: {goal}\n\nAction history:\n" + ("\n".join(hist) if hist else "(none yet)")
        ctx = hdr + f"\n\nCurrent screen ({app_path if where == 'app' else 'Home'}):\n" + "\n".join("- " + s for s in screen)
        lg, H = ask(ctx, QA, order, want_h=(gate == "probe")); calls += 1
        if gate == "contrast":
            lg0, _ = ask(hdr + "\n\nCurrent screen:\n- (not available)", QA, order); calls += 1
            lg = lg + 2.0 * (lg - lg0)
        p = sm(lg); si = order.index(STOP); choice = int(p.argmax())
        if choice == si:
            accept = True; props += 1
            if gate == "verify":
                v, _ = ask(ctx, QV, ["Yes", "No"]); calls += 1; accept = sm(v)[0] >= 0.5
            elif gate == "probe":
                accept = probe.predict_proba(((H - mu) / sd)[None])[0, 1] >= 0.5
            elif gate == "random":
                accept = rng.random() >= reject_rate
            if accept:
                return dict(success=done, stopped=True, steps=step + 1, calls=calls, false_stop=not done, props=props, rej=rej)
            rej += 1; p[si] = -1; choice = int(p.argmax())
        a = order[choice]
        if a.startswith("Open the "):
            where = "app" if a == f"Open the {app} app" else "other"; hist.append(f"Step {len(hist) + 1}: opened {a[9:-4]}")
            if where == "other": where = "home"; hist[-1] += " (wrong app, went back home)"
        elif a == "Navigate back":
            where = "home"; hist.append(f"Step {len(hist) + 1}: navigated back")
        elif a == act:
            tapped += 1; ok = rng.random() >= (P_FAIL if tapped == 1 else 0.2)
            done = done or ok
            claim = ok or rng.random() < P_CLAIM_ON_FAIL
            hist.append(f"Step {len(hist) + 1}: {a.lower()}." + (f" {goal} - done." if claim else " The screen did not change."))
        else:
            hist.append(f"Step {len(hist) + 1}: {a.lower()}.")
    return dict(success=False, stopped=False, steps=MAX_STEPS, calls=calls, false_stop=False, props=props, rej=rej)


goals = [g for i, g in enumerate(G) if (i >= 12) == (FOLD == 0)]
out = {}
for gate in ["none", "verify", "contrast", "probe", "random"]:
    rr = out.get("verify", {}).get("reject_rate", 0.0)
    res = []; rejects = [0, 0]
    for s in range(SEEDS):
        for gi, g in enumerate(goals):
            res.append(dict(episode(g, gate, random.Random(1000 * s + gi + 100000 * FOLD), rr), seed=s, goal=gi + 12 * (1 - FOLD) if FOLD == 0 else gi))
    succ = np.array([r["success"] for r in res]); per_task = succ.reshape(SEEDS, len(goals)).mean(0)
    bs = [np.random.default_rng(i).choice(per_task, len(per_task)).mean() for i in range(2000)]
    calls = np.mean([r["calls"] for r in res])
    out[gate] = dict(success=succ.mean(), ci=[np.percentile(bs, 2.5), np.percentile(bs, 97.5)],
                     false_stop=np.mean([r["false_stop"] for r in res]), timeout=np.mean([not r["stopped"] for r in res]),
                     steps=np.mean([r["steps"] for r in res]), calls_per_episode=calls, calls_per_success=calls * len(res) / max(succ.sum(), 1))
    out[gate]["episodes"] = res
    out[gate]["reject_rate"] = sum(r["rej"] for r in res) / max(sum(r["props"] for r in res), 1)
    print(gate, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[gate].items() if k != 'episodes'}, flush=True)
json.dump(out, open(f"results/e2e_sim_s{SEEDS}_f{FOLD}.json", "w"), indent=1, default=float)
