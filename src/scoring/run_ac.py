"""Score the AndroidControl claim set with one open decision model, in the Jev wire format used by arXiv 2610.13092's T3A
adapter.  Saves probabilities for the operation question (full screen, screen removed), p(yes) for a coupled verify
question, and the hidden state at the model's own decision position for every 2nd layer of the operation question.

usage: run_ac.py {decider|kev|laya} MODEL_ID DATA OUT
"""
import json, sys, time, numpy as np, torch

BACKEND, MODEL, DATA, OUT = sys.argv[1:5]
OPS = {'click': 'Click a UI element.', 'long_press': 'Long press a UI element.', 'input_text': 'Input text in a UI element.',
       'scroll': 'Scroll a UI element or screen.', 'open_app': 'Open an installed app.', 'keyboard_enter': 'Press Enter.',
       'navigate_back': 'Go back.', 'navigate_home': 'Go home.', 'wait': 'Wait for an update.',
       'answer': 'Answer the user explicitly using observed information.', 'complete': 'The requested goal is already achieved.',
       'infeasible': 'The goal cannot be achieved in this environment.'}
Q_OP = {"type": "choice", "instructions": "Following native_t3a_prompt, choose the next single action type to advance the goal.", "criteria": OPS}
Q_VER = {"type": "noul", "instructions": "Judging from the UI elements of the current screen in native_t3a_prompt, has the user goal "
                                         "already been achieved?"}
rows = json.load(open(DATA))
EVERY = 2


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
        from decider.systemone import render_state, render_question
        rq = render_question(q)
        it = build(Example(render_state(state), [Q(rq["question"], rq["options"], 0)]), self.tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768, chat=self.chat)
        ids = torch.tensor([it["ids"]]).cuda(); s = it["slots"][0]
        out = self.m.lm.model(input_ids=ids, output_hidden_states=want_h)
        h = out.last_hidden_state[0, s]
        lg = self.m.cap(h.float() @ self.m.lm.lm_head.weight[self.m.letters].float().T)[: it["nopts"][0]]
        p = torch.softmax(lg / self.T[rq["type"]], -1).cpu().numpy()
        H = torch.stack([x[0, s] for x in out.hidden_states[::EVERY]]).float().cpu().numpy() if want_h else None
        return p, H, len(it["ids"])


class Kev:
    def __init__(self):
        from kev.predictors import LocalPredictor
        from kev.checkpoint import LoadOptions
        self.p = LocalPredictor(MODEL, "cuda", LoadOptions(dtype=torch.bfloat16, merge=True)); self.m = self.p.model; self.tok = self.p.tok
        self.cap = {}
        layers = self.m.lm.layers if hasattr(self.m.lm, "layers") else self.m.lm.model.layers
        for i, l in enumerate(layers):
            if (i + 1) % EVERY == 0:
                l.register_forward_hook(lambda mod, inp, out, i=i: self.cap.__setitem__(i, out[0] if isinstance(out, tuple) else out))

    @torch.no_grad()
    def __call__(self, state, q, want_h):
        from kev.api import SystemOneRequest, to_record
        from kev.model import rows_of
        rec, _ = to_record(SystemOneRequest.model_validate({"state": state, "questions": {"q": q}}))
        enc = self.m.encode(self.tok, rec, max_state=60000, max_branch=62000)
        S, _, brs = rows_of(enc); d = len(S) + brs[0]["decide"]
        self.cap.clear()
        z = self.m.forward_rows_batch([enc])[0][0].float()
        p = torch.softmax(z / self.m.head.temperature, -1).cpu().numpy()
        H = np.stack([self.cap[i][0, d].float().cpu().numpy() for i in sorted(self.cap)]) if want_h else None
        return p, H, len(enc["ids"])


class Laya:
    def __init__(self):
        import laya
        mid, _, sub = MODEL.partition(":")
        self.a = laya.load(mid, device="cuda", subfolder=sub or None); self.cap = []
        enc = self.a.model.encoder if hasattr(self.a, "model") else None
        if enc is not None:
            enc.register_forward_hook(lambda mod, inp, out: self.cap.append(out.last_hidden_state[:, 0].float().mean(0)))

    @torch.no_grad()
    def __call__(self, state, q, want_h):
        self.cap.clear()
        r = self.a.system_one(state, {"q": q}, max_len=8192)
        ans = r["answers"]["q"]
        if q["type"] == "noul":
            py = ans.get("noul", ans.get("probability")); p = np.array([1 - py, py])
        else:
            pr = ans.get("probabilities") or ans.get("probs") or {}
            p = np.array([pr.get(k, 0.0) for k in OPS])
        H = torch.stack(self.cap).mean(0)[None].cpu().numpy() if (want_h and self.cap) else None
        return p, H, 0


M = {"decider": Decider, "kev": Kev, "laya": Laya}[BACKEND]()
P, P0, V, HS, NT = [], [], [], [], []
t0 = time.time()
for i, r in enumerate(rows):
    p, H, n = M({"native_t3a_prompt": r["prompt"]}, Q_OP, True)
    p0, _, _ = M({"native_t3a_prompt": r["prompt_noobs"]}, Q_OP, False)
    v, _, _ = M({"native_t3a_prompt": r["prompt"]}, Q_VER, False)
    P.append(p); P0.append(p0); V.append(v[1]); NT.append(n)
    if H is not None: HS.append(H.astype(np.float16))
    if i % 200 == 0: print(i, f"{time.time() - t0:.0f}s", n, flush=True)
np.savez_compressed(OUT, P=np.array(P), P0=np.array(P0), V=np.array(V), H=np.array(HS) if HS else np.zeros(0), NT=np.array(NT),
                    ops=np.array(list(OPS)))
print("saved", OUT, time.time() - t0)
