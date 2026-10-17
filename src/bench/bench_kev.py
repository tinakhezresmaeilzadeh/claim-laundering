"""EviGate overhead in Kev's serving path (kev.predictors.LocalPredictor, bf16 merged LoRA, shared-prefix rows on the
hybrid Qwen3.5 backbone). The <decide> token is the last token of every question row, so the gate is a forward hook on
layer L reading h[:, -1] and one dot product. Real T3A states, batch 1 (LocalPredictor scores one request at a time).
usage: bench_kev.py ITERS   (run in venv_kev)
"""
import json, sys, numpy as np, torch
from kev.predictors import LocalPredictor
from kev.checkpoint import LoadOptions
sys.path[:0] = ["src/common", "src/data"]
from bench_overhead_common import OPS_Q

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20; LAYER = 20
p = LocalPredictor("jaredpalmer/kev-4b", "cuda", LoadOptions(dtype=torch.bfloat16, merge=True))
layers = p.model.lm.layers if hasattr(p.model.lm, "layers") else p.model.lm.model.layers
d = p.model.lm.config.hidden_size; w = torch.randn(d, device="cuda") * 0.01; G = {}
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    G["s"] = torch.sigmoid(h[:, -1].float() @ w)
rows = json.load(open("results/ac_sub.json"))[:20]
from kev.api import SystemOneRequest, to_record
recs = [to_record(SystemOneRequest.model_validate({"state": {"native_t3a_prompt": r["prompt"]}, "questions": {"op": OPS_Q}}))[0] for r in rows]
encs = [p.model.encode(p.tok, rec, max_state=60000, max_branch=62000) for rec in recs]


@torch.no_grad()
def score(enc):   # Kev's serving miss path for a hybrid backbone: state once, question rows from its cache
    return p.model.probs(enc)


def bench():
    for e in encs[:2]: score(e)
    ts = []
    for k in range(ITERS):
        e = encs[k % len(encs)]
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); score(e); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return float(np.percentile(ts, 50)), float(np.percentile(ts, 99))


base = bench(); hd = layers[LAYER - 1].register_forward_hook(hook); gate = bench(); hd.remove()
res = dict(base_p50=base[0], base_p99=base[1], evigate_p50=gate[0], evigate_p99=gate[1], overhead_p50=100 * (gate[0] / base[0] - 1))
print(json.dumps(res)); json.dump(res, open("results/overhead_kev4b.json", "w"), indent=1)
