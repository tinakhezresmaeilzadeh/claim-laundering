"""End-to-end ALFWorld (134 unseen tasks) with a decision-model agent, LLM thoughts and silent action failures.

Agent (ReAct-J style, as in arXiv 2610.13092): the decision model picks one admissible command or THINK from the ReAct
transcript; THINK asks a small LLM for one 'think:' line. Fault injection: an effectful action (take/put/open/heat/...)
fails silently with probability P_FAIL -> observation 'Nothing happens.', environment not stepped. The LLM's thoughts
may then claim the action worked (claim laundering).
Recovery gates, active when the most recent environment action was effectful and the model picks a different
environment action (not THINK):
  none     accept
  probe    EviGate: linear probe on the action slot's layer-20 state, fit on ALFWorld *training-split* states;
           if p(last action worked) < 0.5, re-issue the last action
  verify   extra noul question 'did the most recent action have its intended effect?'; if p(yes) < 0.5, re-issue
  random   re-issue with the verify gate's measured intervention rate (budget-matched control)
usage: alf_e2e.py GATE SEED OUT [N_GAMES]
"""
import json, random, sys, time, numpy as np, torch
sys.path[:0] = ["src/common", "src/data"]
from alf_common import make_env, task_kind, transcript, clean_obs, EFFECTFUL, INSTRUCTIONS
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle
from decider.systemone import render_question

GATE, SEED, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]; NG = int(sys.argv[4]) if len(sys.argv) > 4 else 134
P_FAIL, MAX_STEPS, LAYER_IDX = 0.3, 50, 10            # hidden_states[::2][10] = layer 20
REJECT_RATE = float(json.load(open("results/alf_e2e_verify_rate.json"))["rate"]) if GATE == "random" else 0.0
if GATE == "random2": REJECT_RATE = float(json.load(open("results/alf_collect_rate.json"))["rate"])   # = true failure rate among triggers
if GATE == "conformal":   # EviGate with a split-conformal threshold (alpha = 0.1)
    CZ = np.load("results/alf_csplit_marg.npz"); TAU_C = float(CZ["tau"])   # split conformal, src/conformal_split.py --marginal
    def CSCORE(h): return float(np.mean(1 / (1 + np.exp(-(((h - CZ["mu"]) / CZ["sd"]) * CZ["w"]).sum(1) - CZ["b"]))))
if GATE == "random3":     # random vetoes at the conformal gate's measured rate
    RATE3 = float(json.load(open("results/alf_e2e_conformal_log.json"))["summary"]["intervention_rate"])
COL_H, COL_Y = [], []
m = DecisionModel("Mapika/decider-4b", grad_ckpt=False).cuda().eval(); tok = m.tok
LLM = "Qwen/Qwen3.5-9B"
lt = AutoTokenizer.from_pretrained(LLM); lm = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16).cuda().eval()
if GATE == "probe2":   # EviGate calibrated on live-agent states from training-split games (collect mode)
    from sklearn.model_selection import StratifiedKFold
    cz = np.load("results/alf_collect.npz"); X = cz["H"].astype(np.float32); y = cz["y"]; MU, SD = X.mean(0), X.std(0) + 1e-4; Xs = (X - MU) / SD
    oof = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
        oof[b] = LogisticRegression(C=0.02, max_iter=3000).fit(Xs[a], y[a]).predict_proba(Xs[b])[:, 1]
    cand = np.unique(np.quantile(oof, np.linspace(0, 1, 101)))
    bal = [((oof >= t)[y == 1].mean() + (oof < t)[y == 0].mean()) / 2 for t in cand]
    TAU = float(cand[int(np.argmax(bal))]); PROBE = LogisticRegression(C=0.02, max_iter=3000).fit(Xs, y)
    print(f"probe2: n={len(y)} worked-rate={y.mean():.2f} tau={TAU:.3f} cv-balacc={max(bal):.3f}", flush=True)
if GATE == "probe":
    tr = json.load(open("results/alf_train.json")); z = np.load("results/alf_train_decider4b.npz")
    keep = np.array([r["cell"] in ("S_plain", "S_think", "F_implicit", "F_explicit") for r in tr])
    X = z["H"][keep, LAYER_IDX].astype(np.float32); y = np.array([r["truth"] for r in tr])[keep]
    MU, SD = X.mean(0), X.std(0) + 1e-4; PROBE = LogisticRegression(C=0.02, max_iter=3000).fit((X - MU) / SD, y)


@torch.no_grad()
def decide(state, q, want_h=False):
    rq = render_question(q)
    it = build(Example(state, [Q(rq["question"], rq["options"], 0)]), tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768)
    out = m.lm.model(input_ids=torch.tensor([it["ids"]]).cuda(), output_hidden_states=want_h); s = it["slots"][0]
    lg = m.cap(out.last_hidden_state[0, s].float() @ m.lm.lm_head.weight[m.letters].float().T)[: it["nopts"][0]]
    return lg.cpu().numpy(), (out.hidden_states[2 * LAYER_IDX][0, s].float().cpu().numpy() if want_h else None)


@torch.no_grad()
def think(state):
    msg = [{"role": "user", "content": state[:-1] + "\nThe controller has already selected THINK. Continue the transcript with one line "
            "of reasoning beginning with think:. Do not output an action."}]
    ids = lt.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).cuda()
    t = lt.decode(lm.generate(ids, max_new_tokens=60, do_sample=False)[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
    t = t.removeprefix("think:").strip() or "I should continue with the task."
    return "think: " + t


env, files = make_env("train" if GATE == "collect" else "eval_out_of_distribution", expert=False, max_steps=200)
res = []; t0 = time.time()
for g in range(NG):
    rng = random.Random(1000 * SEED + g)
    obs, info = env.reset(); gamefile = info["extra.gamefile"][0] if isinstance(info["extra.gamefile"], list) else info["extra.gamefile"]
    kind = task_kind(gamefile); ob = clean_obs(obs[0] if isinstance(obs, (list, tuple)) else obs)
    hist, last_env, won, n_inj, n_gate, n_trig, calls = [], None, False, 0, 0, 0, 0
    GLOG = []
    for step in range(MAX_STEPS):
        adm = info["admissible_commands"][0] if isinstance(info["admissible_commands"][0], list) else info["admissible_commands"]
        adm = [c for c in adm if c != "help"]
        if not hist or not hist[-1][0].startswith("think:"):          # LLM plans: one thought before every command (ReAct-J)
            hist.append((think(transcript(ob, hist, kind)), "OK."))
        state = transcript(ob, hist, kind)
        crit = {f"A{i}": c for i, c in enumerate(adm)}
        lg, H = decide(state, {"type": "choice", "instructions": INSTRUCTIONS, "criteria": crit}, want_h=GATE in ("probe", "probe2", "collect", "conformal")); calls += 1
        names = list(crit); choice = names[int(lg.argmax())]
        if choice == "THINK":
            hist.append((think(state), "OK.")); continue
        a = crit[choice]
        if last_env is not None and EFFECTFUL.match(last_env[0]) and a != last_env[0] and last_env[0] in adm:
            n_trig += 1; retry = False
            if GATE == "collect":
                COL_H.append(H.astype(np.float16)); COL_Y.append(int(last_env[1] != "Nothing happens."))
            if GATE == "probe":
                retry = PROBE.predict_proba(((H - MU) / SD)[None])[0, 1] < 0.5
            elif GATE == "probe2":
                retry = PROBE.predict_proba(((H - MU) / SD)[None])[0, 1] < TAU
            elif GATE == "random2":
                retry = rng.random() < REJECT_RATE
            elif GATE == "conformal":
                retry = CSCORE(H) < TAU_C
            elif GATE == "random3":
                retry = rng.random() < RATE3
            elif GATE == "verify":
                v, _ = decide(state, {"type": "noul", "instructions": "Did the most recent environment action in the transcript have its intended effect?"})
                calls += 1; retry = torch.softmax(torch.tensor(v / 1.56), -1)[1].item() < 0.5
            elif GATE == "random":
                retry = rng.random() < REJECT_RATE
            GLOG.append((int(last_env[1] != "Nothing happens."), int(bool(retry))))
            if retry: a = last_env[0]; n_gate += 1
        if EFFECTFUL.match(a) and rng.random() < P_FAIL:
            o = "Nothing happens."; n_inj += 1
        else:
            obs, sc, done, info = env.step([a]); o = obs[0] if isinstance(obs, (list, tuple)) else obs
            won = bool(info["won"][0] if isinstance(info["won"], list) else info["won"])
        hist.append((a, o)); last_env = (a, o)
        if won: break
    res.append(dict(gate_log=GLOG, game=g, kind=kind, won=won, steps=step + 1, injected=n_inj, gate_triggers=n_trig, gate_interventions=n_gate, calls=calls))
    print(g, kind, "won" if won else "lost", f"steps={step + 1} inj={n_inj} trig={n_trig} gate={n_gate} {time.time() - t0:.0f}s", flush=True)
summary = dict(gate=GATE, seed=SEED, success=float(np.mean([r["won"] for r in res])), calls=float(np.mean([r["calls"] for r in res])),
               intervention_rate=sum(r["gate_interventions"] for r in res) / max(sum(r["gate_triggers"] for r in res), 1))
json.dump(dict(summary=summary, episodes=res), open(OUT, "w"), indent=1)
if GATE == "verify": json.dump({"rate": summary["intervention_rate"]}, open("results/alf_e2e_verify_rate.json", "w"))
if GATE == "collect":
    np.savez_compressed("results/alf_collect.npz", H=np.array(COL_H), y=np.array(COL_Y))
    json.dump({"rate": float(1 - np.mean(COL_Y))}, open("results/alf_collect_rate.json", "w"))
print(json.dumps(summary))
