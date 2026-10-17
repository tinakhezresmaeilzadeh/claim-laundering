"""Score ALFWorld recovery states (build_alf.py) with an open decision model in the Jev format of the paper's ALFWorld
adapter (state = ReAct transcript, choice over commands with ids A0..An + its instruction). Saves the probability of
proceeding / retrying, a coupled verify question, and the decision-position hidden state of every 2nd layer.
usage: run_alf.py {decider|kev} MODEL DATA OUT [INSTR_JSON]   (also used for WebShop with src/common/ws_instr.json)
"""
import json, sys, time, numpy as np, torch
sys.path[:0] = ["src/common", "src/data"]
BACKEND, MODEL, DATA, OUT = sys.argv[1:5]
INSTR = json.load(open(sys.argv[5] if len(sys.argv) > 5 else "src/common/alf_instr.json"))["instructions"]
Q_VER = {"type": "noul", "instructions": "Did the most recent environment action in the transcript have its intended effect?"}
rows = json.load(open(DATA)); EVERY = 2


def q_of(opts):
    return {"type": "choice", "instructions": INSTR, "criteria": {f"A{i}": c for i, c in enumerate(opts)}}


class Decider:
    def __init__(self):
        from decider.model import DecisionModel
        self.m = DecisionModel(MODEL, grad_ckpt=False).cuda().eval(); self.tok = self.m.tok; self.T = {"choice": 1.11, "noul": 1.56}
        from huggingface_hub import hf_hub_download
        from decider.prompt import chat_for
        try: cfg = json.load(open(hf_hub_download(MODEL, "decider_config.json")))
        except Exception: cfg = {}
        self.chat = chat_for(self.tok, cfg)                                   # Gemma decider-12b: chat layout
        tbt = cfg.get("temperature_by_type") or {}
        if tbt: self.T = {"choice": tbt.get("choice", 1.11), "noul": tbt.get("noul", 1.56)}

    @torch.no_grad()
    def __call__(self, state, q, want_h):
        from decider.prompt import build
        from decider.infer import Example, Q, _NoShuffle
        from decider.systemone import render_question
        rq = render_question(q)
        it = build(Example(state, [Q(rq["question"], rq["options"], 0)]), self.tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768, chat=self.chat)
        out = self.m.lm.model(input_ids=torch.tensor([it["ids"]]).cuda(), output_hidden_states=want_h); s = it["slots"][0]
        lg = self.m.cap(out.last_hidden_state[0, s].float() @ self.m.lm.lm_head.weight[self.m.letters].float().T)[: it["nopts"][0]]
        H = torch.stack([x[0, s] for x in out.hidden_states[::EVERY]]).float().cpu().numpy() if want_h else None
        return torch.softmax(lg / self.T[rq["type"]], -1).cpu().numpy(), H


class Kev:
    def __init__(self):
        from kev.predictors import LocalPredictor
        from kev.checkpoint import LoadOptions
        self.p = LocalPredictor(MODEL, "cuda", LoadOptions(dtype=torch.bfloat16, merge=True)); self.m = self.p.model; self.tok = self.p.tok; self.cap = {}
        layers = self.m.lm.layers if hasattr(self.m.lm, "layers") else self.m.lm.model.layers
        for i, l in enumerate(layers):
            if (i + 1) % EVERY == 0: l.register_forward_hook(lambda mod, inp, out, i=i: self.cap.__setitem__(i, out[0] if isinstance(out, tuple) else out))

    @torch.no_grad()
    def __call__(self, state, q, want_h):
        from kev.api import SystemOneRequest, to_record
        from kev.model import rows_of
        rec, _ = to_record(SystemOneRequest.model_validate({"state": state, "questions": {"q": q}}))
        enc = self.m.encode(self.tok, rec, max_state=60000, max_branch=62000); S, _, brs = rows_of(enc); d = len(S) + brs[0]["decide"]
        self.cap.clear(); z = self.m.forward_rows_batch([enc])[0][0].float()
        H = np.stack([self.cap[i][0, d].float().cpu().numpy() for i in sorted(self.cap)]) if want_h else None
        return torch.softmax(z / self.m.head.temperature, -1).cpu().numpy(), H


M = {"decider": Decider, "kev": Kev}[BACKEND](); P, V, HS = [], [], []; t0 = time.time()
for i, r in enumerate(rows):
    p, H = M(r["prompt"], q_of(r["options"]), True); v, _ = M(r["prompt"], Q_VER, False)
    P.append(p); V.append(v[1]); HS.append(H.astype(np.float16))
    if i % 200 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
np.savez_compressed(OUT, P=np.array(P), V=np.array(V), H=np.array(HS))
print("saved", OUT, time.time() - t0)
