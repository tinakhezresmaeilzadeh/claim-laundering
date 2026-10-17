"""Serving-overhead benchmark for decider-4b on real T3A/Jev states (one RTX A6000, bf16, eager PyTorch).

Configurations, per batch of B decision requests:
  base      one forward pass, answer-slot logits                        (the decision model alone)
  evigate   base + gather of the layer-L slot state (forward hook) + linear readout + sigmoid
  verify    base + one extra question row (verify noul) for every request  (separate causal row, as decider's
            independent=True mode runs it; shares nothing with the operation row)
  contrast  base + a second forward pass with the screen removed
Reports per-batch latency p50/p99 (ms) and throughput (decisions/s); CUDA events, warm-up excluded.
usage: bench_overhead.py MODEL N_ITERS
"""
import json, sys, time, numpy as np, torch
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle
from decider.systemone import render_state, render_question

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Mapika/decider-4b"; ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 30
LAYER = 20
sys.path.insert(0, "src")
OPS_Q = {"type": "choice", "instructions": "Following native_t3a_prompt, choose the next single action type to advance the goal.",
         "criteria": {k: v for k, v in [('click', 'Click a UI element.'), ('long_press', 'Long press a UI element.'),
         ('input_text', 'Input text in a UI element.'), ('scroll', 'Scroll a UI element or screen.'), ('open_app', 'Open an installed app.'),
         ('keyboard_enter', 'Press Enter.'), ('navigate_back', 'Go back.'), ('navigate_home', 'Go home.'), ('wait', 'Wait for an update.'),
         ('answer', 'Answer the user explicitly using observed information.'), ('complete', 'The requested goal is already achieved.'),
         ('infeasible', 'The goal cannot be achieved in this environment.')]}}
VER_Q = {"type": "noul", "instructions": "Judging from the UI elements of the current screen in native_t3a_prompt, has the user goal already been achieved?"}

m = DecisionModel(MODEL, grad_ckpt=False).cuda().eval(); tok = m.tok
rows = json.load(open("results/ac_sub.json"))
def item(prompt, q):
    rq = render_question(q)
    return build(Example(render_state({"native_t3a_prompt": prompt}), [Q(rq["question"], rq["options"], 0)]), tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768)
W = torch.randn(m.lm.config.text_config.hidden_size if hasattr(m.lm.config, "text_config") else m.lm.config.hidden_size, device="cuda") * 0.01
MU = torch.zeros_like(W); SD = torch.ones_like(W)
CAP = {}
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    CAP["h"] = h[CAP["bi"], CAP["slot"]]


@torch.no_grad()
def forward(items):
    T = max(len(it["ids"]) for it in items)
    ids = torch.full((len(items), T), tok.pad_token_id, device="cuda"); att = torch.zeros_like(ids)
    for b, it in enumerate(items):
        ids[b, :len(it["ids"])] = torch.tensor(it["ids"], device="cuda"); att[b, :len(it["ids"])] = 1
    CAP["bi"] = torch.arange(len(items), device="cuda"); CAP["slot"] = torch.tensor([it["slots"][0] for it in items], device="cuda")
    h = m.lm.model(input_ids=ids, attention_mask=att).last_hidden_state[CAP["bi"], CAP["slot"]]
    return torch.softmax(m.cap(h.float() @ m.lm.lm_head.weight[m.letters].float().T), -1)


def run(cfg, B):
    sel = rows[:B]
    a = [item(r["prompt"], OPS_Q) for r in sel]
    v = [item(r["prompt"], VER_Q) for r in sel]; a0 = [item(r["prompt_noobs"], OPS_Q) for r in sel]
    hd = m.lm.model.layers[LAYER - 1].register_forward_hook(hook) if cfg == "evigate" else None
    def once():
        p = forward(a)
        if cfg == "evigate":
            s = torch.sigmoid(((CAP["h"].float() - MU) / SD) @ W); _ = (s < 0.5)
        elif cfg == "verify":
            _ = forward(v)
        elif cfg == "contrast":
            _ = forward(a0)
    for _ in range(3): once()
    ts = []
    for _ in range(ITERS):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); once(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    if hd: hd.remove()
    ts = np.array(ts); return dict(p50=float(np.percentile(ts, 50)), p99=float(np.percentile(ts, 99)), thr=float(B / (np.mean(ts) / 1000)),
                                   tokens=int(np.mean([len(it["ids"]) for it in a])))


res = {}
for B in (1, 8, 32):
    for cfg in ("base", "evigate", "verify", "contrast"):
        torch.cuda.empty_cache()
        try:
            r = run(cfg, B)
        except torch.OutOfMemoryError:
            r = dict(p50=float("nan"), p99=float("nan"), thr=float("nan"), tokens=0)
        res[f"{cfg}_B{B}"] = r
        base = res[f"base_B{B}"]["p50"]
        print(f"B={B:2d} {cfg:9s} p50={r['p50']:8.1f}ms p99={r['p99']:8.1f}ms thr={r['thr']:6.1f}/s  overhead_p50={100 * (r['p50'] / base - 1):+6.2f}%  avg_tokens={r['tokens']}", flush=True)
json.dump(res, open(f"results/overhead_{MODEL.split('/')[-1]}.json", "w"), indent=1)
