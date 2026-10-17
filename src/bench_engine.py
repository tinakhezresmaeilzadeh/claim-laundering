"""EviGate inside decider's production Engine (torch.compile + CUDA graphs for T<=2048, fused-conv eager above).

EviEngine registers a hook on layer L that computes the gate score for every position inside the same compiled /
graph-captured forward and returns it as one extra column next to the option logits ([B, T, K+1]); the answer-slot
read is unchanged, the gate is one more gathered value. Benchmarks score_items latency of Engine vs EviEngine on
(a) real T3A states (eager path, ~3-4k tokens) and (b) short synthetic decision states (graph path).
usage: bench_engine.py ITERS
"""
import json, sys, numpy as np, torch
from decider.engine import Engine
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle
from decider.systemone import render_state, render_question
from decider.model import cap_logits
import torch.nn.functional as F

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 30; LAYER = 20
sys.path.insert(0, "src")
from bench_overhead_common import OPS_Q


class EviEngine(Engine):
    def __init__(self, *a, layer=LAYER, **k):
        self._gate_buf = {}
        super().__init__(*a, **k)
        d = self.m.lm.config.get_text_config().hidden_size
        self.w = (torch.randn(d, device=self.dev) * 0.01); self.mu = torch.zeros(d, device=self.dev); self.sd = torch.ones(d, device=self.dev)
        self.core.layers[layer - 1].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self._gate_buf["s"] = torch.sigmoid(((h.float() - self.mu) / self.sd) @ self.w)      # [B, T]

    def _fwd_eager(self, ids):
        h = self.core(input_ids=ids, use_cache=False).last_hidden_state
        lg = cap_logits(self.m, F.linear(h, self.W).float())
        return torch.cat([lg, self._gate_buf["s"][..., None]], -1)                            # [B, T, K+1]


def items_for(prompts, tok):
    rq = render_question(OPS_Q)
    return [build(Example(render_state({"native_t3a_prompt": p}), [Q(rq["question"], rq["options"], 0)]), tok, _NoShuffle(),
                  max_options=255, max_ctx_tokens=32768) for p in prompts]


def bench(eng, items, B):
    its = items[:B]
    for _ in range(3): eng.score_items(its)
    ts = []
    for _ in range(ITERS):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); eng.score_items(its); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 99))


real = [r["prompt"] for r in json.load(open("results/ac_sub.json"))[:32]]
short = [r["ctx"] for r in json.load(open("results/data.json"))[:32]]
res = {}
for name, cls in (("base", Engine), ("evigate", EviEngine)):
    eng = cls("Mapika/decider-4b")
    for regime, prompts in (("t3a_long", real), ("short", short)):
        its = items_for(prompts, eng.tok)
        for B in (1, 8, 32):
            torch.cuda.empty_cache()
            try: p50, p99 = bench(eng, its, B)
            except torch.OutOfMemoryError: p50 = p99 = float("nan")
            res[f"{name}_{regime}_B{B}"] = dict(p50=p50, p99=p99, thr=B / (p50 / 1000), tokens=int(np.mean([len(i["ids"]) for i in its[:B]])))
            print(f"{name:8s} {regime:9s} B={B:2d} p50={p50:8.2f}ms p99={p99:8.2f}ms thr={B / (p50 / 1000):7.1f}/s tokens={res[f'{name}_{regime}_B{B}']['tokens']}", flush=True)
    del eng; torch.cuda.empty_cache()
for k in [k for k in res if k.startswith("evigate")]:
    b = res[k.replace("evigate", "base")]; print(f"overhead {k[8:]:14s} p50 {100 * (res[k]['p50'] / b['p50'] - 1):+.2f}%")
json.dump(res, open("results/overhead_engine.json", "w"), indent=1)
