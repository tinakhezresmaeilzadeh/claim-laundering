"""Evaluate decider-4b with/without the CIT adapter on the same states.
Domains: phones (held-out apps of results/ac.json only), ALFWorld (results/alf_eval.json), WebShop (results/ws_eval.json):
  wrong      : terminal argmax on Logged+claim (failed step reported as successful)
  done_ok    : terminal argmax on Done / Done+claim
  bal_acc    : (done_ok + not-terminal rate on Logged / Logged+claim) / 2
  effects    : paired d log p(terminal): observation (Done - Logged), logged step (Logged - Honest), claim (Logged+claim - Logged)
  auroc      : head log p(terminal), worked vs failed (Honest excluded), all and claimed-only (Done+claim vs Logged+claim)
General ability: accuracy on decider's held-out tasks hwu64 and dbpedia_l2 (first 400 examples each).
usage: cit_eval.py {base|ADAPTER_DIR} OUT.json
"""
import hashlib, json, sys, numpy as np, torch
sys.path[:0] = ["src/common", "src/data"]
from sklearn.metrics import roc_auc_score
from decider.model import DecisionModel
from decider.prompt import build
from decider.infer import Example, Q, _NoShuffle
from decider.systemone import render_state, render_question
from bench_overhead_common import OPS_Q

WHO, OUT = sys.argv[1], sys.argv[2]
m = DecisionModel("Mapika/decider-4b", grad_ckpt=False).cuda().eval(); tok = m.tok
if WHO != "base":
    from peft import PeftModel
    m.lm = PeftModel.from_pretrained(m.lm, WHO).merge_and_unload().eval()
is_test_app = lambda a: int(hashlib.md5(a.encode()).hexdigest(), 16) % 5 == 0
CMAP = {"S_plain": "D_plain", "S_think": "D_explicit", "F_honest": "N_honest", "F_implicit": "N_implicit", "F_explicit": "N_explicit"}


@torch.no_grad()
def lp(ctx, q, opts):
    it = build(Example(ctx, [Q(q, opts, 0)]), tok, _NoShuffle(), max_options=255, max_ctx_tokens=32768)
    h = m.lm.model(input_ids=torch.tensor([it["ids"]]).cuda()).last_hidden_state[0, it["slots"][0]]
    return torch.log_softmax(m.cap(h.float() @ m.lm.lm_head.weight[m.letters].float().T)[: it["nopts"][0]], -1).cpu().numpy()


def domain(name, rows):
    rec = []
    for r in rows:
        if name == "phones":
            rq = render_question(OPS_Q); T = list(OPS_Q["criteria"]).index("complete")
            l = lp(render_state({"native_t3a_prompt": r["prompt"]}), rq["question"], rq["options"])
            key = str(r["ep"])
        else:
            instr = json.load(open("src/common/alf_instr.json" if name == "alfworld" else "src/common/ws_instr.json"))["instructions"]
            rq = render_question({"type": "choice", "instructions": instr, "criteria": {f"A{i}": c for i, c in enumerate(r["options"])}})
            l = lp(r["prompt"], rq["question"], rq["options"]); T = r["proceed"]
            key = f"{r['game']}_{r['step']}" if name == "alfworld" else str(r["goal"])
        rec.append(dict(key=key, cell=CMAP.get(r["cell"], r["cell"]), lpT=float(l[T]), term=int(np.argmax(l)) == T))
    by = {}
    for x in rec: by.setdefault(x["key"], {})[x["cell"]] = x
    pair = lambda a, b: float(np.mean([v[a]["lpT"] - v[b]["lpT"] for v in by.values() if a in v and b in v]))
    rate = lambda c: float(np.mean([x["term"] for x in rec if x["cell"] == c]))
    nh = [x for x in rec if x["cell"] != "N_honest"]; y = np.array([x["cell"].startswith("D_") for x in nh]); s = np.array([x["lpT"] for x in nh])
    cl = np.array([x["cell"] in ("D_explicit", "N_explicit") for x in nh])
    done_ok = (rate("D_plain") + rate("D_explicit")) / 2; fail_term = (rate("N_implicit") + rate("N_explicit")) / 2
    return dict(n=len(rec), wrong=rate("N_explicit"), wrong_logged=rate("N_implicit"), done_ok=done_ok, bal_acc=(done_ok + 1 - fail_term) / 2,
                effects=dict(observation=pair("D_plain", "N_implicit"), logged=pair("N_implicit", "N_honest"), claim=pair("N_explicit", "N_implicit")),
                auroc=dict(all=float(roc_auc_score(y, s)), claimed=float(roc_auc_score(y[cl], s[cl]))))


res = {}
res["phones"] = domain("phones", [r for r in json.load(open("results/ac.json")) if is_test_app(r["app"])]); print("phones", res["phones"], flush=True)
res["alfworld"] = domain("alfworld", json.load(open("results/alf_eval.json"))); print("alfworld", res["alfworld"], flush=True)
res["webshop"] = domain("webshop", json.load(open("results/ws_eval.json"))); print("webshop", res["webshop"], flush=True)
from decider.data import core as D
import decider.data.tasks_heldout  # noqa: F401  (registers the held-out tasks)
for t in ["hwu64", "dbpedia_l2"]:
    _, exs = D.TASKS[t]["loader"](); exs = exs[:400]
    res[t] = float(np.mean([int(np.argmax(lp(e.context, e.qs[0].text, e.qs[0].options)) == e.qs[0].gold) for e in exs])); print(t, res[t], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
