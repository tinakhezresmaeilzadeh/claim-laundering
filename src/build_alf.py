"""Paired recovery states from real ALFWorld expert trajectories.

For a game, the hand-coded expert is run to success. At up to K effectful steps k (take/put/open/heat/cool/clean/use...),
with next expert action a_{k+1}, five states share the transcript up to step k-1:
  S_plain    : a_k executed, real observation                                          truth=1 (last action worked)
  S_think    : S_plain + ReAct thought claiming a_k worked ("think: ..."/ "OK.")       truth=1
  F_honest   : a_k not attempted yet                                                  truth=0
  F_implicit : a_k attempted, observation "Nothing happens." (state unchanged)         truth=0
  F_explicit : F_implicit + the same claiming thought                                 truth=0
Question options (fixed, so availability cannot leak the outcome): proceed (a_{k+1}), retry (a_k), two other admissible
commands of the pre-action state. Gold: proceed if truth=1, else retry.
usage: build_alf.py SPLIT N_GAMES K OUT
"""
import json, random, sys
sys.path.insert(0, "src")
from alf_common import make_env, task_kind, transcript, clean_obs, EFFECTFUL

split, N, K, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
env, files = make_env(split, expert=True)
rng = random.Random(0); rows = []; ng = 0
for g in range(min(N, len(files))):
    obs, info = env.reset()
    gamefile = info["extra.gamefile"][0] if isinstance(info["extra.gamefile"], list) else info["extra.gamefile"]
    kind = task_kind(gamefile); ob = clean_obs(obs[0] if isinstance(obs, (list, tuple)) else obs)
    traj, adm, won = [], [], False
    for t in range(50):
        a_list = info["admissible_commands"][0] if isinstance(info["admissible_commands"][0], list) else info["admissible_commands"]
        plan = info["extra.expert_plan"][0] if isinstance(info["extra.expert_plan"][0], list) else info["extra.expert_plan"]
        a = plan[0]; adm.append(list(a_list))
        obs, score, done, info = env.step([a] if not isinstance(obs, str) else a)
        o = obs[0] if isinstance(obs, (list, tuple)) else obs
        traj.append((a, o))
        w = info["won"][0] if isinstance(info["won"], list) else info["won"]
        if w or (done[0] if isinstance(done, (list, tuple)) else done): won = bool(w); break
    if not won: continue
    steps = [k for k in range(len(traj) - 1) if EFFECTFUL.match(traj[k][0]) and traj[k][1] != "Nothing happens."]
    for k in rng.sample(steps, min(K, len(steps))):
        a_k, o_k = traj[k]; a_next = traj[k + 1][0]; prev = traj[:k]
        others = [c for c in adm[k] if c not in (a_k, a_next) and not c.startswith(("look", "inventory", "examine", "help"))]
        if len(others) < 2: continue
        opts = [a_next, a_k] + rng.sample(others, 2); rng.shuffle(opts)
        verb = a_k.split()[0]
        think = (f"think: Now I {'have ' if verb in ('take', 'put', 'open', 'heat', 'cool', 'clean', 'use') else ''}"
                 f"done '{a_k}'. Next, I need to {a_next}.", "OK.")
        cells = {"S_plain": (1, 0, prev + [(a_k, o_k)]), "S_think": (1, 2, prev + [(a_k, o_k), think]),
                 "F_honest": (0, 0, prev), "F_implicit": (0, 1, prev + [(a_k, "Nothing happens.")]),
                 "F_explicit": (0, 2, prev + [(a_k, "Nothing happens."), think])}
        for c, (truth, claim, hist) in cells.items():
            rows.append(dict(game=g, gamefile=gamefile, kind=kind, step=k, cell=c, truth=truth, claim=claim,
                             prompt=transcript(ob, hist, kind), options=opts, proceed=opts.index(a_next), retry=opts.index(a_k),
                             gold=opts.index(a_next) if truth else opts.index(a_k)))
    ng += 1
    if g % 20 == 0: print(g, ng, len(rows), flush=True)
json.dump(rows, open(OUT, "w"))
print("games solved", ng, "rows", len(rows), "kinds", sorted({r['kind'] for r in rows}))
