"""LLM-written step summaries with T3A's own summarization prompt (android_world/agents/t3a.py), replacing the templated
claim of the final step.

For each episode of the subset, the final action N is summarised twice by an instruction-tuned LLM:
  worked : before = screen N-1, after = screen N      (the action had its effect)
  failed : before = screen N-1, after = screen N-1    (silent failure: the screen did not change)
Steps 1..N-1 keep their template summaries. New states:
  D_llm      : screen N,   history 1..N-1 + worked summary      gold: done
  N_llm_fail : screen N-1, history 1..N-1 + failed summary      gold: not done
  N_honest   : screen N-1, history 1..N-1                       gold: not done
Also records whether the failed summary itself asserts success (keyword heuristic + an LLM yes/no check).
usage: llm_summaries.py LLM OUT
"""
import json, re, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path[:0] = ["src/common", "src/data"]
from build_ac import TEMPLATE, prompt as t3a_prompt

LLM, OUT = sys.argv[1], sys.argv[2]
AW = "ext/jevemp/src/third_party/android_world/android_world"
src = open(f"{AW}/agents/t3a.py").read(); ns = {}
exec(re.search(r"PROMPT_PREFIX = \(.*?\n\)\n", src, re.S).group(0), ns)
exec(re.search(r"SUMMARIZATION_PROMPT_TEMPLATE = \(.*?\n\)\n", src, re.S).group(0), ns)
SUM = ns["SUMMARIZATION_PROMPT_TEMPLATE"]

tok = AutoTokenizer.from_pretrained(LLM); m = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.bfloat16).cuda().eval()


@torch.no_grad()
def chat(text, max_new=120):
    msgs = [{"role": "user", "content": text}]
    try: ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
    except TypeError: ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = ids.cuda() if torch.is_tensor(ids) else ids["input_ids"].cuda()
    out = m.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]


def ui_of(p):
    a = p.index("current screen:\n") + len("current screen:\n"); b = p.index("\n", p.rfind("UI element", a)) if "UI element" in p[a:] else a
    return p[a:b]


rows = json.load(open("results/ac_sub.json")); by = {}
for r in rows: by.setdefault(r["ep"], {})[r["cell"]] = r
out = []; n_claim = 0
for k, (ep, c) in enumerate(by.items()):
    goal = c["D_plain"]["goal"]
    hist_full = c["N_implicit"]["prompt"].split("Here is a history of what you have done so far:\n")[1].split("\n\nHere is a list")[0].split("\n")
    last = hist_full[-1]; prev = hist_full[:-1]
    m_act = re.match(r"Step \d+: (.*?) in order to (.*)\.$", last)
    action, reason = (m_act.group(1), m_act.group(2)) if m_act else (last, "continue the task")
    uiP, uiN = ui_of(c["N_implicit"]["prompt"]), ui_of(c["D_plain"]["prompt"])
    s_ok = chat(SUM.format(goal=goal, before_elements=uiP, after_elements=uiN, action=action, reason=reason))
    s_fail = chat(SUM.format(goal=goal, before_elements=uiP, after_elements=uiP, action=action, reason=reason))
    judge = chat(f"A phone agent wrote this step summary: \"{s_fail}\"\nDoes the summary state or imply that the action succeeded "
                 f"or that the goal \"{goal}\" is now accomplished? Answer yes or no.", 3).lower().startswith("yes")
    n_claim += judge
    step = len(prev) + 1
    h_ok, h_fail = prev + [f"Step {step}: {s_ok}"], prev + [f"Step {step}: {s_fail}"]
    out.append(dict(ep=ep, app=c["D_plain"]["app"], goal=goal, cell="D_llm", truth=1, claim=1, prompt=t3a_prompt(goal, h_ok, uiN),
                    prompt_noobs=t3a_prompt(goal, h_ok, ""), summary=s_ok))
    out.append(dict(ep=ep, app=c["D_plain"]["app"], goal=goal, cell="N_llm_fail", truth=0, claim=int(judge), prompt=t3a_prompt(goal, h_fail, uiP),
                    prompt_noobs=t3a_prompt(goal, h_fail, ""), summary=s_fail))
    out.append(dict(c["N_honest"], cell="N_honest"))
    if k % 20 == 0: print(k, "| ok:", s_ok[:110], "| fail:", s_fail[:110], "| claims:", judge, flush=True)
json.dump(out, open(OUT, "w"))
print(f"failed-step summaries that claim success: {n_claim}/{len(by)} = {n_claim / len(by):.3f}")
