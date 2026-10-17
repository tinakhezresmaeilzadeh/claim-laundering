"""Answer-slot activation patching (as mechanism.py) on the ALFWorld / WebShop paired states, decider-4b.
Target metric: log p(proceed). Pairs (target <- source), restoration = (lp_patched - lp_target) / (lp_source - lp_target):
  claim    : F_explicit <- F_implicit   (removes the claiming thought)
  implicit : F_implicit <- F_honest     (removes the logged failed attempt)
  evidence : F_implicit <- S_plain      (injects the 'worked' observation)
usage: mechanism_gen.py DATA INSTR_JSON N_PAIRS OUT   (model via env MECH_MODEL, default Mapika/decider-4b)
"""
import json, sys, numpy as np, torch
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle
from decider.systemone import render_question

DATA, INSTR, N, OUT = sys.argv[1], json.load(open(sys.argv[2]))["instructions"], int(sys.argv[3]), sys.argv[4]
WS = {"D_plain": "S_plain", "D_explicit": "S_think", "N_honest": "F_honest", "N_implicit": "F_implicit", "N_explicit": "F_explicit"}
rows = json.load(open(DATA)); by = {}
for r in rows:
    c = WS.get(r["cell"], r["cell"]); by.setdefault((r.get("game", r.get("goal")), r.get("step", 0)), {})[c] = r
keys = list(by)[:N]
import os
MODEL = os.environ.get("MECH_MODEL", "Mapika/decider-4b")
m = DecisionModel(MODEL, grad_ckpt=False).cuda().eval(); tok = m.tok
LAYERS = m.lm.model.layers if hasattr(m.lm.model, "layers") else m.lm.model.language_model.layers   # Qwen3.5 / Gemma-4
L = len(LAYERS); ST = {}
from huggingface_hub import hf_hub_download
from decider.prompt import chat_for
try: _cfg = json.load(open(hf_hub_download(MODEL, "decider_config.json")))
except Exception: _cfg = {}
CHAT = chat_for(tok, _cfg)


def hook_for(k):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if ST.get("mode") == "read": ST["cache"][k] = h[0, ST["slot"]].clone()
        elif ST.get("mode") == "write" and k == ST["cur"]: h[0, ST["slot"]] = ST["src"]
        return out
    return hook


for k, l in enumerate(LAYERS): l.register_forward_hook(hook_for(k))


@torch.no_grad()
def lp(r):
    q = render_question({"type": "choice", "instructions": INSTR, "criteria": {f"A{i}": c for i, c in enumerate(r["options"])}})
    it = build(Example(r["prompt"], [Q(q["question"], q["options"], 0)]), tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768, chat=CHAT)
    ST["slot"] = it["slots"][0]
    h = m.lm.model(input_ids=torch.tensor([it["ids"]]).cuda()).last_hidden_state[0, it["slots"][0]]
    lg = m.cap(h.float() @ m.lm.lm_head.weight[m.letters].float().T)[: it["nopts"][0]]
    return float(torch.log_softmax(lg, -1)[r["proceed"]])


layers = list(range(0, L, 2)) + [L - 1]
PAIRS = {"claim": ("F_explicit", "F_implicit"), "implicit": ("F_implicit", "F_honest"), "evidence": ("F_implicit", "S_plain")}
res = {k: [] for k in PAIRS}
for n, key in enumerate(keys):
    for name, (tgt, src) in PAIRS.items():
        if tgt not in by[key] or src not in by[key]: continue
        ST.update(mode="read", cache={}); lp_s = lp(by[key][src]); cache = ST["cache"]
        ST.update(mode=None); lp_t = lp(by[key][tgt])
        if abs(lp_s - lp_t) < 0.2: continue
        curve = []
        for l in layers:
            ST.update(mode="write", cur=l, src=cache[l]); curve.append((lp(by[key][tgt]) - lp_t) / (lp_s - lp_t))
        ST.update(mode=None); res[name].append(curve)
    if n % 20 == 0: print(n, {k: len(v) for k, v in res.items()}, flush=True)
out = {"layers": layers}
for name, v in res.items():
    a = np.array(v); out[name] = dict(n=len(a), mean=a.mean(0).tolist() if len(a) else [], se=(a.std(0) / np.sqrt(max(len(a), 1))).tolist() if len(a) else [])
    if len(a): print(name, f"n={len(a)}", " ".join(f"L{l}:{x:+.2f}" for l, x in zip(layers, a.mean(0))))
json.dump(out, open(OUT, "w"), indent=1)
