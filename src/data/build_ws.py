"""Paired completion states from the real WebShop text environment (synthetic attribute goals over 1,000 products).

For a goal with required options, the agent searches the product title, opens the target item and clicks every required
option (real environment steps). Five states (decision: 'buy now' vs. select the first required option):
  D_plain    : options clicked and applied (page shows '[clicked button]' / 'You have clicked ...')   truth=1
  D_explicit : D_plain + think line claiming the options are selected                              truth=1
  N_honest   : item page, options not yet clicked                                                  truth=0
  N_implicit : option clicks logged, but the page is unchanged (silent failure)                    truth=0
  N_explicit : N_implicit + the same claiming think line                                           truth=0
Options (fixed): buy now, first required option, description, back to search.
usage: build_ws.py START N OUT
"""
import json, random, sys
sys.path[:0] = ["src/common", "src/data"]
from ws_common import make_env

START, N, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
env = make_env(); env.reset(session=0); G = env.server.goals; rng = random.Random(START)
HEAD = "WebShop\n"


def transcript(instr, steps):
    body = "".join(f"Action: {a}\nObservation:\n{(o if k == len(steps) - 1 else o[:1200])}\n\n" for k, (a, o) in enumerate(steps))
    return HEAD + f"Instruction:\n{instr}\n\n" + body + "Action:"


rows = []; k = START
while len(set(r["goal"] for r in rows)) < N and k < len(G):
    g = G[k]; k += 1
    opts = g.get("goal_options") or {}
    vals = list(opts.values()) if isinstance(opts, dict) else list(opts)
    if not vals or any(c in v for v in vals for c in '"&#%+'): continue
    out = env.reset(session=k - 1); instr = g["instruction_text"]
    title = env.server.product_item_dict[g["asin"]]["Title"]
    s_obs, *_ = env.step(f"search[{title[:80]}]")
    if g["asin"].lower() not in env.get_available_actions()["clickables"]: continue
    p0, *_ = env.step(f"click[{g['asin'].lower()}]")
    cl = env.get_available_actions()["clickables"]
    if not all(v in cl for v in vals): continue
    clicks, obs = [], p0
    for v in vals:
        obs, *_ = env.step(f"click[{v}]"); clicks.append((f"click[{v}]", obs))
    if "[clicked button]" not in obs: continue
    base = [(f"search[{title[:80]}]", s_obs), (f"click[{g['asin'].lower()}]", p0)]
    claim = (f"think[I have selected {', '.join(vals)} as required. Now I can buy it.]", "OK.")
    failed = [(a, p0) for a, _ in clicks]
    cells = {"D_plain": (1, 0, base + clicks), "D_explicit": (1, 2, base + clicks + [claim]), "N_honest": (0, 0, base),
             "N_implicit": (0, 1, base + failed), "N_explicit": (0, 2, base + failed + [claim])}
    options = ["buy now", vals[0], "description", "back to search"]; rng.shuffle(options)
    for c, (truth, cl_, steps) in cells.items():
        rows.append(dict(goal=k - 1, cell=c, truth=truth, claim=cl_, prompt=transcript(instr, steps), options=options,
                         proceed=options.index("buy now"), retry=options.index(vals[0]),
                         gold=options.index("buy now") if truth else options.index(vals[0]), category=env.server.product_item_dict[g["asin"]].get("category", "")))
    if len(rows) % 100 == 0: print(k, len(rows), flush=True)
json.dump(rows, open(OUT, "w")); print("goals", len(set(r["goal"] for r in rows)), "rows", len(rows))
