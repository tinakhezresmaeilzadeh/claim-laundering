"""Score the paired claim-vs-observation states with the hosted Jev model (TypeSafe, via OpenRouter System One API).
Same request format as the open models: Android = run_ac.py (T3A state, 12-operation choice + verify noul); ALFWorld /
WebShop = run_alf.py (transcript state, choice over the fixed options with ids A0..An + verify noul). Jev returns
probabilities only, so this measures behaviour (no probes). Resumable: finished rows are kept in OUT.jsonl.
usage: run_jev.py {ac|alf|ws} DATA OUT_PREFIX [N] [WORKERS]
"""
import json, os, sys, time, threading, requests
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "src")

KIND, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9; W = int(sys.argv[5]) if len(sys.argv) > 5 else 8
KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL, MODEL = "https://openrouter.ai/api/v1/systemone", "typesafe/jev-1.13"
rows = json.load(open(DATA))[:N]
if KIND == "ac":
    from bench_overhead_common import OPS_Q
    Q_VER = {"type": "noul", "instructions": "Judging from the UI elements of the current screen in native_t3a_prompt, has the user goal already been achieved?"}
else:
    INSTR = json.load(open("src/alf_instr.json" if KIND == "alf" else "src/ws_instr.json"))["instructions"]
    Q_VER = {"type": "noul", "instructions": "Did the most recent environment action in the transcript have its intended effect?"}
lock = threading.Lock(); outf = OUT + ".jsonl"
done = {json.loads(l)["i"] for l in open(outf)} if os.path.exists(outf) else set()


def call(state, questions):
    for attempt in range(6):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {KEY}"}, json={"model": MODEL, "state": state, "questions": questions}, timeout=120)
            if r.status_code == 200: return r.json()
            if r.status_code in (408, 409, 429) or r.status_code >= 500: time.sleep(2 ** attempt); continue
            return {"error": r.status_code, "text": r.text[:300]}
        except requests.RequestException as e:
            time.sleep(2 ** attempt); err = str(e)
    return {"error": "retries", "text": err if "err" in dir() else ""}


def work(i):
    r = rows[i]
    if KIND == "ac":
        state = {"native_t3a_prompt": r["prompt"]}; qa = OPS_Q
    else:
        state = r["prompt"]; qa = {"type": "choice", "instructions": INSTR, "criteria": {f"A{j}": c for j, c in enumerate(r["options"])}}
    a = call(state, {"op": qa}); v = call(state, {"ver": Q_VER})
    rec = {"i": i, "op": a.get("answers", {}).get("op"), "ver": v.get("answers", {}).get("ver"),
           "cost": (a.get("usage", {}).get("cost") or 0) + (v.get("usage", {}).get("cost") or 0),
           "tokens": (a.get("usage", {}).get("input_tokens") or 0), "model": a.get("model"),
           "error": a.get("error") or v.get("error")}
    with lock:
        with open(outf, "a") as f: f.write(json.dumps(rec) + "\n")
    return rec


todo = [i for i in range(len(rows)) if i not in done]; t0 = time.time(); cost = 0.0; n = 0; errs = 0
print(f"{KIND}: {len(rows)} rows, {len(done)} already done, {len(todo)} to go", flush=True)
with ThreadPoolExecutor(W) as ex:
    for rec in ex.map(work, todo):
        n += 1; cost += rec["cost"]; errs += bool(rec["error"])
        if n % 100 == 0 or n == len(todo): print(f"{n}/{len(todo)} cost=${cost:.4f} errors={errs} {time.time() - t0:.0f}s", flush=True)
print("done", OUT, f"cost=${cost:.4f}")
