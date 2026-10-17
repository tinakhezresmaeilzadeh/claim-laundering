"""Best-of-N with a decision model as the verifier (WebShop pilot).
For each goal, sample K full trajectories from the same agent as ws_e2e.py (Qwen3.5-4B thoughts/searches sampled at
temperature 0.8, decider-4b picks clicks, option clicks fail silently with probability 0.3, no gate). Every trajectory that
buys is then scored by decider-4b as a verifier on the transcript up to and including 'click[buy now]' (the post-purchase
page, which shows the reward, is excluded):
  noul question VERIFY_Q -> p(yes)  and the layer-20 verifier-slot state (for an EviGate-style probe verifier, fit offline
  across goals). Logged per trajectory: reward, success (reward == 1), whether the options were really selected, and whether
  the agent's own thoughts claimed the options were selected.
usage: ws_bon.py K OUT.json FIRST_GOAL N_GOALS
"""
import json, random, re, sys, time, numpy as np, torch
sys.path[:0] = ["src/common", "src/data"]
from ws_common import make_env
from transformers import AutoModelForCausalLM, AutoTokenizer
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle
from decider.systemone import render_question

K, OUT, G0, NG = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
P_FAIL, MAX_STEPS, LIDX, TEMP = 0.3, 15, 10, 0.8
INSTR = json.load(open("src/common/ws_instr.json"))["instructions"]
VERIFY_Q = "Did the agent complete the task: is the purchased product the requested one, with every option required by the instruction selected?"
NAV = {"back to search", "< prev", "next >", "description", "features", "reviews", "buy now"}
CLAIM = re.compile(r"(?i)(selected|chosen|chose|picked|set)\b.*|now i can buy|ready to (buy|purchase)|all (required )?options")
m = DecisionModel("Mapika/decider-4b", grad_ckpt=False).cuda().eval(); tok = m.tok
LLM = "Qwen/Qwen3.5-4B"; lt = AutoTokenizer.from_pretrained(LLM); lm = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16).cuda().eval()


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
    out = lm.generate(ids, max_new_tokens=n, do_sample=True, temperature=TEMP, top_p=0.95)
    return lt.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]


env = make_env(); env.reset(session=0); res = []; H = []; t0 = time.time()
for gi in range(G0, G0 + NG):
    instr = env.server.goals[gi]["instruction_text"]
    gopt = env.server.goals[gi].get("goal_options") or {}; gv = list(gopt.values()) if isinstance(gopt, dict) else list(gopt)
    for k in range(K):
        torch.manual_seed(100003 * gi + k); rng = random.Random(7919 * gi + k)
        out = env.reset(session=gi); page = out[0] if isinstance(out, tuple) else out
        steps, reward, bought, claimed, opts_ok = [], 0.0, False, False, False
        for t in range(MAX_STEPS):
            av = env.get_available_actions()
            if av["has_search_bar"]:
                q = llm(transcript(instr, steps), "Write the next action as search[keywords] using the key attributes of the instruction. Output only the action.", 40)
                a = q if q.startswith("search[") and q.endswith("]") else f"search[{instr[:80]}]"
            else:
                th = llm(transcript(instr, steps), "Write one short reasoning line beginning with think[ and ending with ] about what to do next on this page.")
                th = th if th.startswith("think[") else f"think[{th}]"; steps.append((th, "OK.")); claimed |= bool(CLAIM.search(th))
                cl = av["clickables"]; lg, _ = decide(transcript(instr, steps), {"type": "choice", "instructions": INSTR, "criteria": {f"A{i}": c for i, c in enumerate(cl)}})
                a_c = cl[int(np.argmax(lg))]; a = f"click[{a_c}]"
                if a_c == "buy now":
                    sel = set(env.server.user_sessions[env.session]["options"].values()); opts_ok = all(v in sel for v in gv)
                    vstate = transcript(instr, steps + [(a, "")])[: -len("Observation:\n\n\nAction:")].rstrip()
                if a_c not in NAV and not a_c.startswith("b0") and rng.random() < P_FAIL:
                    steps.append((a, page)); continue                    # silent failure: page unchanged
            obs, r, done, info = env.step(a); page = obs; steps.append((a, obs))
            if done: reward, bought = float(r), True; break
        v_yes, h = None, None
        if bought:
            v, h = decide(vstate, {"type": "noul", "instructions": VERIFY_Q}, want_h=True)
            v_yes = float(torch.softmax(torch.tensor(v / 1.56), -1)[1]); H.append(h.astype(np.float16))
        res.append(dict(goal=gi, k=k, reward=reward, success=reward >= 1.0, bought=bought, opts_ok=bool(opts_ok), claimed=claimed,
                        verifier_yes=v_yes, h_idx=(len(H) - 1) if bought else None, steps=t + 1))
    g = [r for r in res if r["goal"] == gi]
    print(gi, f"bought {sum(r['bought'] for r in g)}/{K} success {sum(r['success'] for r in g)}/{K} "
              f"unearned-claim {sum(r['bought'] and r['claimed'] and not r['opts_ok'] for r in g)} {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(OUT, "w")); np.save(OUT.replace(".json", "_H.npy"), np.array(H) if H else np.zeros((0, 2560), np.float16))
print("done")
