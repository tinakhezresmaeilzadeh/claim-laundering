"""Print the paper's main numbers from the included summaries (results/summaries). No GPU, no models, no data download.
usage: python scripts/show_results.py
"""
import json, os
R = os.path.join(os.path.dirname(__file__), "..", "results", "summaries")
J = lambda f: json.load(open(os.path.join(R, f)))

print("Table 2 - which input moves the decision (paired effect on log p of stop / proceed / buy)")
print(f"  {'domain / model':26s} {'Obs.':>7s} {'Logged':>7s} {'Claim':>7s}")
for k, v in J("table2_ci.json").items():
    print(f"  {k:26s} {v['obs'][0]:+7.2f} {v['logged'][0]:+7.2f} {v['claim'][0]:+7.2f}")

print("\nTable 5 - task success (%) of live agents")
phone = {g: J("e2e_v2_pooled.json")[g]["success"] for g in ("none", "verify", "probe")}
f0, f1 = J("e2e_v2_s30_f0_rate.json"), J("e2e_v2_s30_f1_rate.json")
phone["random"] = (f0["random4"]["success"] + f1["random4"]["success"]) / 2
alf = {g: J(f"alf_e2e_{f}.json")["summary"]["success"] for g, f in
       [("none", "none"), ("verify", "verify"), ("offline", "probe"), ("probe", "probe2"), ("random", "random2")]}
ws = {g: J(f"ws_e2e_{f}.json")["summary"]["success"] for g, f in
      [("none", "none"), ("verify", "verify"), ("offline", "probe"), ("probe", "probe2"), ("random", "random2")]}
rows = [("No gate", "none"), ("Second question", "verify"), ("EviGate, fit on offline contrast sets", "offline"),
        ("EviGate, fit on target-domain states", "probe"), ("Random veto, matched rate", "random")]
print(f"  {'condition':40s} {'Phones':>7s} {'ALFWorld':>9s} {'WebShop':>8s}")
for name, k in rows:
    p = f"{100 * phone[k]:7.1f}" if k in phone else f"{'--':>7s}"
    print(f"  {name:40s} {p} {100 * alf[k]:9.1f} {100 * ws[k]:8.1f}")

print("\nTable 6 - fine-tuning decider-4b (wrong = failed steps that claim success and still end in a terminal decision)")
runs = [("base", "cit_eval_base.json"), ("phones", "cit_eval_tuned.json"), ("all 3, 1k each", "cit_eval_all.json"),
        ("all 3, all phones", "cit_eval_all_fullphones.json")]
print(f"  {'fine-tuned on':20s} {'Phones':>7s} {'ALFWorld':>9s} {'WebShop':>8s} {'HWU64':>6s} {'DBpedia':>8s}")
for name, f in runs:
    d = J(f)
    print(f"  {name:20s} {100 * d['phones']['wrong']:6.1f}% {100 * d['alfworld']['wrong']:8.1f}% {100 * d['webshop']['wrong']:7.1f}%"
          f" {100 * d['hwu64'] + 1e-9:6.1f} {100 * d['dbpedia_l2'] + 1e-9:8.1f}")   # round half up, as in the paper
