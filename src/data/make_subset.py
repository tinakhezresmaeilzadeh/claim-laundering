"""The 200-episode phone subset (results/ac_sub.json) used for the larger models, hosted Jev, LLM summaries, patching and
removing the record: the episodes listed in data/ac_sub_episodes.json, taken from results/ac.json.
usage: make_subset.py results/ac.json data/ac_sub_episodes.json results/ac_sub.json
"""
import json, sys
rows = json.load(open(sys.argv[1])); keep = json.load(open(sys.argv[2])); order = {e: i for i, e in enumerate(keep)}
sub = sorted([r for r in rows if r["ep"] in order], key=lambda r: order[r["ep"]])
json.dump(sub, open(sys.argv[3], "w")); print(len(order), "episodes", len(sub), "rows")
