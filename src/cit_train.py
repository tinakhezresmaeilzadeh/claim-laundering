"""Claim-invariant tuning (CIT): LoRA fine-tuning of decider-4b on paired states (phones by default; see DOMAINS),
so that the same claim with a different observation gives a different decision.
Phones split: apps hashed into 5 buckets, bucket 0 = held-out test apps (never trained on). ALFWorld/WebShop test sets are
never trained on (their training sets use different games / goals).
Loss per state (operation question, 12 options, terminal = 'complete'):
  terminal BCE : -[y log p(complete) + (1-y) log(1 - p(complete))],  y = 1 for Done/Done+claim, 0 for Honest/Logged/Logged+claim
  keep         : KL(base || tuned) over the non-terminal options (renormalised), so the model keeps how it ranks other operations
usage: cit_train.py OUT_DIR [LAMBDA_KEEP] [DOMAINS] [PER_DOMAIN]
  DOMAINS: comma list of phones,alfworld,webshop (default phones). ALFWorld/WebShop use their TRAINING sets
  (results/alf_train.json: ALFWorld train split, disjoint from the valid_unseen test games; results/ws_train.json: goals
  disjoint from the test goals). Terminal option: 'complete' (phones) or the proceed / 'buy now' option (r["proceed"]).
  PER_DOMAIN: random subsample of states per domain (default: all); or per domain in DOMAINS, e.g. phones:all,alfworld:1000.
"""
import hashlib, json, math, random, sys, time, numpy as np, torch
sys.path.insert(0, "src")
from peft import LoraConfig, get_peft_model
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle
from decider.systemone import render_state, render_question
from bench_overhead_common import OPS_Q

OUT = sys.argv[1]; LAM = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
DOMS = (sys.argv[3] if len(sys.argv) > 3 else "phones").split(","); PER = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
# per-domain sizes: "phones:all,alfworld:1000" (a domain without ":n" uses PER)
SIZE = {d.split(":")[0]: (10 ** 9 if d.split(":")[1] == "all" else int(d.split(":")[1])) if ":" in d else PER for d in DOMS}
DOMS = [d.split(":")[0] for d in DOMS]
LR, ACCUM, EPOCHS, WARM, SEED = 1e-4, 8, 1, 20, 0
is_test_app = lambda a: int(hashlib.md5(a.encode()).hexdigest(), 16) % 5 == 0
torch.manual_seed(SEED); random.seed(SEED)
m = DecisionModel("Mapika/decider-4b", grad_ckpt=True).cuda(); tok = m.tok
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
pm = get_peft_model(m.lm, cfg); m.lm.enable_input_require_grads(); pm.print_trainable_parameters()
items = []
for dom in DOMS:
    if dom == "phones":
        rows = [r for r in json.load(open("results/ac.json")) if not is_test_app(r["app"])]
        rq = render_question(OPS_Q); Tc = list(OPS_Q["criteria"]).index("complete")
    else:
        rows = json.load(open("results/alf_train.json" if dom == "alfworld" else "results/ws_train.json"))
        instr = json.load(open("src/alf_instr.json" if dom == "alfworld" else "src/ws_instr.json"))["instructions"]
    rows = random.Random(SEED).sample(rows, min(SIZE[dom], len(rows)))
    for r in rows:
        if dom == "phones": ctx, q, opts, t = render_state({"native_t3a_prompt": r["prompt"]}), rq["question"], rq["options"], Tc
        else:
            qq = render_question({"type": "choice", "instructions": instr, "criteria": {f"A{i}": c for i, c in enumerate(r["options"])}})
            ctx, q, opts, t = r["prompt"], qq["question"], qq["options"], r["proceed"]
        it = build(Example(ctx, [Q(q, opts, 0)]), tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768)
        items.append((it["ids"], it["slots"][0], it["nopts"][0], int(r["cell"][0] in "DS"), t))
    print(f"{dom}: {len(rows)} states", flush=True)
print(f"train states {len(items)}; done-rate {np.mean([i[3] for i in items]):.2f}", flush=True)


def logits_of(ids, slot, n):
    h = m.lm.model(input_ids=torch.tensor([ids]).cuda()).last_hidden_state[0, slot]
    return m.cap(h.float() @ m.lm.lm_head.weight[m.letters].float().T)[:n]


params = [p for p in pm.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
total = math.ceil(len(items) * EPOCHS / ACCUM); step = 0; t0 = time.time(); acc = []
lr_at = lambda s: LR * s / WARM if s < WARM else LR * 0.5 * (1 + math.cos(math.pi * (s - WARM) / max(1, total - WARM)))
rest = lambda n, T: [i for i in range(n) if i != T]
m.train()
for ep in range(EPOCHS):
    random.Random(SEED + ep).shuffle(items)
    for k, (ids, slot, n, y, T) in enumerate(items):
        with torch.no_grad(), pm.disable_adapter():
            base = logits_of(ids, slot, n)
        lg = logits_of(ids, slot, n); lp = torch.log_softmax(lg, -1)
        lpT = lp[T]; lnotT = torch.logsumexp(lp[rest(n, T)], 0)
        bce = -(lpT if y else lnotT)
        q = torch.softmax(base[rest(n, T)], -1); lp_rest = torch.log_softmax(lg[rest(n, T)], -1)
        keep = (q * (torch.log(q + 1e-12) - lp_rest)).sum()
        loss = (bce + LAM * keep) / ACCUM; loss.backward(); acc.append((bce.item(), keep.item()))
        if (k + 1) % ACCUM == 0:
            for g in opt.param_groups: g["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(); step += 1
            if step % 20 == 0:
                a = np.array(acc[-20 * ACCUM:]); print(f"step {step}/{total} bce {a[:, 0].mean():.3f} keep {a[:, 1].mean():.4f} "
                                                      f"{(time.time() - t0) / 60:.1f}min mem {torch.cuda.max_memory_allocated() / 1e9:.0f}GB", flush=True)
pm.save_pretrained(OUT); print("saved", OUT, flush=True)
