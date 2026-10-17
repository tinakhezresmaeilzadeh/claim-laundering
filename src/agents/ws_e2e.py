"""End-to-end WebShop (synthetic attribute goals, 1,000 products) with an LLM-plans / decision-model-acts agent and
silent option-click failures; completion gates on 'buy now'.

Each step: on the search page the LLM writes 'search[...]'; elsewhere the LLM writes one think[...] line and the decision
model picks one clickable. An option click fails silently with probability P_FAIL (page unchanged). The episode ends at
'buy now' (reward from WebShop's attribute/option/price matcher) or after MAX_STEPS.
Gates when the model picks 'buy now':
  none | probe (EviGate, layer 20, fit on WebShop training goals 3000+) | verify (noul: 'have all options required by
  the instruction been selected on the current page?') | random (verify's measured veto rate).
A veto takes the best non-'buy now' clickable instead.
usage: ws_e2e.py GATE SEED OUT N_GOALS [FIRST_GOAL]   (GATE also: collect, probe2)
"""
import json, random, sys, time, numpy as np, torch
sys.path[:0] = ["src/common", "src/data"]
from ws_common import make_env
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle
from decider.systemone import render_question

GATE, SEED, OUT, NG = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
P_FAIL, MAX_STEPS, LIDX = 0.3, 15, 10
INSTR = json.load(open("src/common/ws_instr.json"))["instructions"]
VERIFY = "Have all options required by the instruction (such as size or color) been selected on the current page?"
NAV = {"back to search", "< prev", "next >", "description", "features", "reviews", "buy now"}
REJ = float(json.load(open("results/ws_e2e_verify_rate.json"))["rate"]) if GATE == "random" else 0.0
if GATE == "random2": REJ = float(json.load(open("results/ws_e2e_probe2.json"))["summary"]["veto_rate"]); GATE_R2 = True
m = DecisionModel("Mapika/decider-4b", grad_ckpt=False).cuda().eval(); tok = m.tok
LLM = "Qwen/Qwen3.5-4B"; lt = AutoTokenizer.from_pretrained(LLM); lm = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16).cuda().eval()
if GATE == "conformal":   # EviGate with a split-conformal threshold (alpha = 0.1)
    CZ = np.load("results/ws_csplit_marg.npz"); TAU_C = float(CZ["tau"])   # split conformal, src/conformal_split.py --marginal
    def CSCORE(h): return float(np.mean(1 / (1 + np.exp(-(((h - CZ["mu"]) / CZ["sd"]) * CZ["w"]).sum(1) - CZ["b"]))))
if GATE == "random3":     # random vetoes at the conformal gate's measured rate
    RATE3 = float(json.load(open("results/ws_e2e_conformal_log.json"))["summary"]["veto_rate"])
COL_H, COL_Y, COL_BUY = [], [], []
if GATE == "probe2":   # EviGate calibrated on live-agent states (collect mode, goals disjoint from the test goals)
    from sklearn.model_selection import StratifiedKFold
    cz = np.load("results/ws_collect.npz"); X = cz["H"].astype(np.float32); y = cz["y"]; MU, SD = X.mean(0), X.std(0) + 1e-4
    Xs = (X - MU) / SD; oof = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
        oof[b] = LogisticRegression(C=0.02, max_iter=3000).fit(Xs[a], y[a]).predict_proba(Xs[b])[:, 1]
    buy = cz["buy"].astype(bool); cand = np.unique(np.quantile(oof[buy], np.linspace(0, 1, 101)))
    bal = [((oof[buy] >= t)[y[buy] == 1].mean() + (oof[buy] < t)[y[buy] == 0].mean()) / 2 for t in cand]
    TAU = float(cand[int(np.argmax(bal))]); PROBE = LogisticRegression(C=0.02, max_iter=3000).fit(Xs, y)
    print(f"probe2: n={len(y)} buy-states={buy.sum()} pos-rate={y.mean():.2f} tau={TAU:.3f} cv-balacc={max(bal):.3f}", flush=True)
if GATE == "probe":
    tr = json.load(open("results/ws_train.json")); z = np.load("results/ws_train_decider4b.npz")
    keep = np.array([r["cell"] != "N_honest" for r in tr]); X = z["H"][keep, LIDX].astype(np.float32)
    y = np.array([r["truth"] for r in tr])[keep]; MU, SD = X.mean(0), X.std(0) + 1e-4
    PROBE = LogisticRegression(C=0.02, max_iter=3000).fit((X - MU) / SD, y)


def transcript(instr, steps):
    body = "".join(f"Action: {a}\nObservation:\n{(o if k == len(steps) - 1 else o[:1200])}\n\n" for k, (a, o) in enumerate(steps))
    return f"WebShop\nInstruction:\n{instr}\n\n" + body + "Action:"


@torch.no_grad()
def decide(state, q, want_h=False):
    rq = render_question(q)
    it = build(Example(state, [Q(rq["question"], rq["options"], 0)]), tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768)
    out = m.lm.model(input_ids=torch.tensor([it["ids"]]).cuda(), output_hidden_states=want_h); s = it["slots"][0]
    lg = m.cap(out.last_hidden_state[0, s].float() @ m.lm.lm_head.weight[m.letters].float().T)[: it["nopts"][0]]
    return lg.cpu().numpy(), (out.hidden_states[2 * LIDX][0, s].float().cpu().numpy() if want_h else None)


@torch.no_grad()
def llm(state, ask, n=60):
    msg = [{"role": "user", "content": state + "\n\n" + ask}]
    ids = lt.apply_chat_template(msg, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).cuda()
    return lt.decode(lm.generate(ids, max_new_tokens=n, do_sample=False)[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]


env = make_env(); env.reset(session=0); res = []; t0 = time.time()
G0 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
for gi in range(G0, G0 + NG):
    rng = random.Random(1000 * SEED + gi); out = env.reset(session=gi); obs = out[0] if isinstance(out, tuple) else out
    instr = env.server.goals[gi]["instruction_text"]; steps, reward, bought, calls, trig, veto, inj = [], 0.0, False, 0, 0, 0, 0
    GLOG = []
    page = obs
    for t in range(MAX_STEPS):
        av = env.get_available_actions()
        if av["has_search_bar"]:
            q = llm(transcript(instr, steps), "Write the next action as search[keywords] using the key attributes of the instruction. Output only the action.", 40)
            a = q if q.startswith("search[") and q.endswith("]") else f"search[{instr[:80]}]"
        else:
            th = llm(transcript(instr, steps), "Write one short reasoning line beginning with think[ and ending with ] about what to do next on this page.")
            steps.append((th if th.startswith("think[") else f"think[{th}]", "OK."))
            cl = av["clickables"]; crit = {f"A{i}": c for i, c in enumerate(cl)}
            lg, H = decide(transcript(instr, steps), {"type": "choice", "instructions": INSTR, "criteria": crit}, want_h=GATE in ("probe", "probe2", "collect", "conformal")); calls += 1
            order = list(np.argsort(-lg)); a_c = cl[order[0]]
            if GATE == "collect" and "buy now" in cl:
                gopt = env.server.goals[gi].get("goal_options") or {}; gv = list(gopt.values()) if isinstance(gopt, dict) else list(gopt)
                sel = set(env.server.user_sessions[env.session]["options"].values())
                COL_H.append(H.astype(np.float16)); COL_Y.append(int(all(v in sel for v in gv))); COL_BUY.append(int(a_c == "buy now"))
            if a_c == "buy now":
                trig += 1; v_ = False
                if GATE == "probe": v_ = PROBE.predict_proba(((H - MU) / SD)[None])[0, 1] < 0.5
                elif GATE == "probe2": v_ = PROBE.predict_proba(((H - MU) / SD)[None])[0, 1] < TAU
                elif GATE == "verify":
                    v, _ = decide(transcript(instr, steps), {"type": "noul", "instructions": VERIFY}); calls += 1
                    v_ = torch.softmax(torch.tensor(v / 1.56), -1)[1].item() < 0.5
                elif GATE in ("random", "random2"): v_ = rng.random() < REJ
                elif GATE == "conformal": v_ = CSCORE(H) < TAU_C
                elif GATE == "random3": v_ = rng.random() < RATE3
                _gv = env.server.goals[gi].get("goal_options") or {}; _gv = list(_gv.values()) if isinstance(_gv, dict) else list(_gv)
                _sel = set(env.server.user_sessions[env.session]["options"].values()); GLOG.append((int(all(v in _sel for v in _gv)), int(bool(v_))))
                if v_: veto += 1; a_c = cl[order[1]]
            a = f"click[{a_c}]"
            if a_c not in NAV and not a_c.startswith("b0") and rng.random() < P_FAIL:
                steps.append((a, page)); inj += 1; continue          # silent failure: page unchanged
        obs, r, done, info = env.step(a); page = obs; steps.append((a, obs))
        if done: reward, bought = float(r), True; break
    res.append(dict(gate_log=GLOG, goal=gi, reward=reward, success=reward >= 1.0, bought=bought, steps=t + 1, calls=calls, triggers=trig, vetoes=veto, injected=inj))
    print(gi, f"reward={reward:.2f} bought={bought} steps={t + 1} inj={inj} trig={trig} veto={veto} {time.time() - t0:.0f}s", flush=True)
summ = dict(gate=GATE, success=float(np.mean([r["success"] for r in res])), score=float(np.mean([r["reward"] for r in res])),
            calls=float(np.mean([r["calls"] for r in res])), veto_rate=sum(r["vetoes"] for r in res) / max(sum(r["triggers"] for r in res), 1))
json.dump(dict(summary=summ, episodes=res), open(OUT, "w"), indent=1)
if GATE == "collect": np.savez_compressed("results/ws_collect.npz", H=np.array(COL_H), y=np.array(COL_Y), buy=np.array(COL_BUY))
if GATE == "verify": json.dump({"rate": summ["veto_rate"]}, open("results/ws_e2e_verify_rate.json", "w"))
print(json.dumps(summ))
