"""Where does the claim take over? Layer-wise answer-slot activation patching on decider-4b (real AndroidControl subset).

For paired states that differ in one factor, run the target state and replace the answer-slot residual at layer l with
the source state's, then measure how much of the log p(complete) difference moves:
  claim    : target N_explicit, source N_implicit (same screen; removes the explicit claim at the slot)
  implicit : target N_implicit, source N_honest   (same screen; removes the logged-but-ineffective action)
  evidence : target N_implicit, source D_plain    (same history; injects the done screen)
restoration(l) = (lp_patched - lp_target) / (lp_source - lp_target), averaged over episodes with |gap| > 0.2.
Also measures attention mass from the answer slot to the history span vs the screen span in full-attention layers.
usage: mechanism.py N_EPISODES   (model via env MECH_MODEL, default Mapika/decider-4b)
"""
import json, sys, numpy as np, torch
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle
from decider.systemone import render_state, render_question
sys.path[:0] = ["src/common", "src/data"]
from bench_overhead_common import OPS_Q

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
import os
MODEL = os.environ.get("MECH_MODEL", "Mapika/decider-4b")
m = DecisionModel(MODEL, grad_ckpt=False).cuda().eval(); tok = m.tok
LAYERS = m.lm.model.layers if hasattr(m.lm.model, "layers") else m.lm.model.language_model.layers   # Qwen3.5 / Gemma-4
from huggingface_hub import hf_hub_download
from decider.prompt import chat_for
try: _cfg = json.load(open(hf_hub_download(MODEL, "decider_config.json")))
except Exception: _cfg = {}
CHAT = chat_for(tok, _cfg)
L = len(LAYERS); rq = render_question(OPS_Q); C = list(OPS_Q["criteria"]).index("complete")
rows = json.load(open("results/ac_sub.json")); by = {}
for r in rows: by.setdefault(r["ep"], {})[r["cell"]] = r
eps = list(by)[:N]
ST = {}


def item(p):
    return build(Example(render_state({"native_t3a_prompt": p}), [Q(rq["question"], rq["options"], 0)]), tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768, chat=CHAT)


def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if ST.get("mode") == "read": ST["cache"][ST["layer"]] = h[0, ST["slot"]].clone()
    elif ST.get("mode") == "write" and ST["layer"] == ST["cur"]: h[0, ST["slot"]] = ST["src"]
    return out


hooks = [l.register_forward_hook(lambda mod, i, o, k=k: (ST.__setitem__("layer", k), hook(mod, i, o))[1]) for k, l in enumerate(LAYERS)]


@torch.no_grad()
def lp(it):
    ST["slot"] = it["slots"][0]
    h = m.lm.model(input_ids=torch.tensor([it["ids"]]).cuda()).last_hidden_state[0, it["slots"][0]]
    lg = m.cap(h.float() @ m.lm.lm_head.weight[m.letters].float().T)[: it["nopts"][0]]
    return float(torch.log_softmax(lg, -1)[C])


layers = list(range(0, L, 2)) + [L - 1]
PAIRS = {"claim": ("N_explicit", "N_implicit"), "implicit": ("N_implicit", "N_honest"), "evidence": ("N_implicit", "D_plain")}
res = {k: [] for k in PAIRS}
for n, ep in enumerate(eps):
    for name, (tgt, src) in PAIRS.items():
        it_t, it_s = item(by[ep][tgt]["prompt"]), item(by[ep][src]["prompt"])
        ST.update(mode="read", cache={}); lp_s = lp(it_s); cache = ST["cache"]
        ST.update(mode=None); lp_t = lp(it_t)
        if abs(lp_s - lp_t) < 0.2: continue
        curve = []
        for l in layers:
            ST.update(mode="write", cur=l, src=cache[l]); curve.append((lp(it_t) - lp_t) / (lp_s - lp_t))
        ST.update(mode=None); res[name].append(curve)
    if n % 10 == 0: print(n, {k: len(v) for k, v in res.items()}, flush=True)

out = {"layers": layers}
for name, v in res.items():
    a = np.array(v); out[name] = dict(n=len(a), mean=a.mean(0).tolist(), se=(a.std(0) / np.sqrt(max(len(a), 1))).tolist())
    print(name, f"n={len(a)}", " ".join(f"L{l}:{x:+.2f}" for l, x in zip(layers, a.mean(0))))
json.dump(out, open(os.environ.get("MECH_OUT", "results/mechanism_patching.json"), "w"), indent=1)
