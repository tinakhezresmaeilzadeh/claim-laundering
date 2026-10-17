"""Story-blind re-read (offline feasibility, decider-4b). For each paired state:
  full pass   -> log p(terminal option), and the layer-20 decision-position state (for the EviGate probe baseline);
  blind pass  -> re-run layers l..L-1 on the sequence with the agent's self-report tokens DROPPED (original positions kept),
                 starting from the full pass's hidden states at the input of layer l. Dropping (not masking) is required:
                 Qwen3.5 has 24 linear-attention (recurrent) layers, which an attention mask would not hide anything from.
Self-report spans (only text the agent wrote; observations stay):
  phones   : the 'Step k: ...' lines of the action history
  alfworld : the agent's '> action' and '> think: ...' lines of the current task (few-shot examples untouched)
  webshop  : the 'Action: ...' lines (search, clicks, thoughts)
usage: story_blind.py {phones|alfworld|webshop} OUT.json [N_GROUPS]   (OUT.npz gets the probe features)
       story_blind.py selftest
"""
import json, re, sys, numpy as np, torch
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle
from decider.systemone import render_state, render_question
from transformers.masking_utils import create_causal_mask
sys.path[:0] = ["src/common", "src/data"]

import os
LS = [int(x) for x in os.environ.get("SB_LAYERS", "12,16,20,24,28").split(",")]; PL = 20   # re-read start layers (0 = record removed from the input); probe layer

m = DecisionModel("Mapika/decider-4b", grad_ckpt=False).cuda().eval(); tok = m.tok; TM = m.lm.model; L = len(TM.layers)
PRE = "Context:\n"


def report_spans(dom, ctx):
    """Character spans (in ctx) of the agent's own text."""
    if dom == "phones":                            # ctx is JSON-escaped: newlines are the two characters '\' 'n'
        a = ctx.index("Here is a history of what you have done so far:\\n") + len("Here is a history of what you have done so far:\\n")
        b = ctx.index("\\n\\nHere is a list of descriptions", a)
        return [(a, b)]
    if dom == "alfworld":
        t = ctx.rindex("Here is the task.")
        return [(t + mm.start(), t + mm.end()) for mm in re.finditer(r"(?m)^> .+$", ctx[t:])]
    return [(mm.start(), mm.end()) for mm in re.finditer(r"(?m)^Action: .+$", ctx)]


def item(dom, r):
    if dom == "phones":
        from bench_overhead_common import OPS_Q
        ctx = render_state({"native_t3a_prompt": r["prompt"]}); rq = render_question(OPS_Q)
        term = list(OPS_Q["criteria"]).index("complete")
    else:
        instr = json.load(open("src/common/alf_instr.json" if dom == "alfworld" else "src/common/ws_instr.json"))["instructions"]
        ctx = r["prompt"]; rq = render_question({"type": "choice", "instructions": instr,
                                                 "criteria": {f"A{i}": c for i, c in enumerate(r["options"])}})
        term = r["proceed"]
    it = build(Example(ctx, [Q(rq["question"], rq["options"], 0)]), tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768)
    enc = tok(PRE + ctx, add_special_tokens=False, return_offsets_mapping=True)
    assert enc["input_ids"] == it["ids"][:len(enc["input_ids"])], "context tokenisation mismatch"
    drop = np.zeros(len(it["ids"]), bool)
    for a, b in report_spans(dom, ctx):
        a, b = a + len(PRE), b + len(PRE)
        for t, (x, y) in enumerate(enc["offset_mapping"]):
            if x < b and y > a: drop[t] = True
    assert it["slots"][0] == len(it["ids"]) - 1 and not drop[-1]
    return it, drop, term


def logp(h, n):
    lg = m.cap(h.float() @ m.lm.lm_head.weight[m.letters].float().T)[:n]
    return torch.log_softmax(lg, -1)


@torch.no_grad()
def run(it, drop, layers=LS):
    ids = torch.tensor([it["ids"]]).cuda(); n = it["nopts"][0]; cache = {}
    hooks = [TM.layers[l].register_forward_pre_hook(lambda mod, a, kw, l=l: cache.__setitem__(l, (a[0] if a else kw["hidden_states"])),
                                                     with_kwargs=True) for l in set(layers) | {PL}]
    full = logp(TM(input_ids=ids).last_hidden_state[0, -1], n)
    for h in hooks: h.remove()
    keep = torch.tensor(np.where(~drop)[0], device="cuda"); blind = {}
    for l in layers:
        hs = cache[l][:, keep]
        pos = keep.view(1, 1, -1).expand(3, 1, -1)                     # original positions for (M)RoPE (text: all 3 equal)
        pe = TM.rotary_emb(hs, pos)
        cm = create_causal_mask(config=TM.config, inputs_embeds=hs, attention_mask=None, past_key_values=None, position_ids=None)
        for i in range(l, L):
            hs = TM.layers[i](hs, position_embeddings=pe, attention_mask=cm if TM.config.layer_types[i] == "full_attention" else None,
                              position_ids=None, past_key_values=None, use_cache=False)
        blind[l] = logp(TM.norm(hs)[0, -1], n)
    return full, blind, cache[PL][0, -1].float().cpu().numpy()


if sys.argv[1] == "selftest":                       # with nothing dropped the re-read must reproduce the full pass
    for dom, f in [("phones", "results/ac_sub.json"), ("alfworld", "results/alf_eval.json"), ("webshop", "results/ws_eval.json")]:
        r = json.load(open(f))[7]; it, drop, term = item(dom, r)
        full, blind, _ = run(it, np.zeros_like(drop))
        full2, blind2, _ = run(it, drop)
        print(dom, "tokens", len(drop), "dropped", int(drop.sum()), "| max |full - blind(nothing dropped)| =",
              max(float((full - b).abs().max()) for b in blind.values()), "| terminal lp full %.3f blind@20 %.3f" % (full2[term], blind2[20][term]))
        txt = tok.decode([t for t, d in zip(it["ids"], drop) if d]); print("   dropped text sample:", repr(txt[:160]), "...", repr(txt[-120:]))
    sys.exit()

DOM, OUT = sys.argv[1], sys.argv[2]; NG = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
rows = json.load(open({"phones": "results/ac_sub.json", "alfworld": "results/alf_eval.json", "webshop": "results/ws_eval.json"}[DOM]))
WS = {"D_plain": "S_plain", "D_explicit": "S_think", "N_honest": "F_honest", "N_implicit": "F_implicit", "N_explicit": "F_explicit"}
grp = lambda r: str(r["ep"]) if DOM == "phones" else (f"{r['game']}_{r['step']}" if DOM == "alfworld" else str(r["goal"]))
groups = list(dict.fromkeys(grp(r) for r in rows))[:NG]; keepg = set(groups)
res, H = [], []
for k, r in enumerate([r for r in rows if grp(r) in keepg]):
    it, drop, term = item(DOM, r); full, blind, h = run(it, drop)
    res.append(dict(group=grp(r), app=r.get("app", r.get("game", r.get("goal"))), cell=WS.get(r["cell"], r["cell"]), truth=int(r["truth"]),
                    term=term, n_tok=len(drop), n_drop=int(drop.sum()), lp_full=full.tolist(), lp_blind={str(l): b.tolist() for l, b in blind.items()}))
    H.append(h.astype(np.float16))
    if k % 50 == 0: print(DOM, k, flush=True)
json.dump(res, open(OUT, "w")); np.savez_compressed(OUT.replace(".json", ".npz"), H=np.stack(H))
print("done", DOM, len(res))
