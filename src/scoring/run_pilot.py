"""Pilot: does decider-4b trust history claims over screen evidence, and does its hidden state know better than its head?

Saves per row: action-question logits (full ctx, screen-ablated ctx), verify-question logits, and the action-slot hidden
state at every layer.
"""
import json, sys, torch, numpy as np
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Mapika/decider-4b"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/decider4b"
QA = "Which action should the agent take next to accomplish the goal?"
QV = "Does the current screen show that the goal has already been accomplished?"

m = DecisionModel(MODEL, grad_ckpt=False).cuda().eval()
tok = m.tok
rows = json.load(open("results/data.json"))


def item(ctx, q, opts):
    return build(Example(ctx, [Q(q, opts, 0)]), tok, _NoShuffle(), max_options=255)


@torch.no_grad()
def run(items, want_h=False):
    T = max(len(it["ids"]) for it in items)
    ids = torch.full((len(items), T), tok.pad_token_id); att = torch.zeros_like(ids)
    for b, it in enumerate(items):
        ids[b, :len(it["ids"])] = torch.tensor(it["ids"]); att[b, :len(it["ids"])] = 1
    out = m.lm.model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=want_h)
    slot = torch.tensor([it["slots"][0] for it in items]).cuda(); bi = torch.arange(len(items)).cuda()
    h = out.last_hidden_state[bi, slot]
    lg = m.cap(h.float() @ m.lm.lm_head.weight[m.letters].float().T)
    n = [it["nopts"][0] for it in items]
    lgs = [lg[i, :n[i]].cpu().numpy() for i in range(len(items))]
    H = torch.stack([hs[bi, slot] for hs in out.hidden_states], 1).float().cpu().numpy() if want_h else None   # [B,L+1,D]
    return lgs, H


A, A0, V, HS = [], [], [], []
B = 16
for s in range(0, len(rows), B):
    rs = rows[s:s + B]
    la, H = run([item(r["ctx"], QA, r["options"]) for r in rs], want_h=True)
    l0, _ = run([item(r["ctx_noobs"], QA, r["options"]) for r in rs])
    lv, _ = run([item(r["ctx"], QV, ["Yes", "No"]) for r in rs])
    A += la; A0 += l0; V += lv; HS.append(H.astype(np.float16))
    print(s, flush=True)
np.savez_compressed(OUT + ".npz", A=np.array(A, dtype=object), A0=np.array(A0, dtype=object), V=np.array(V), H=np.concatenate(HS),
                    allow_pickle=True)
print("saved", OUT)
