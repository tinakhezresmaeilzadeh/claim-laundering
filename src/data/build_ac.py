"""Realistic claim-vs-evidence set from AndroidControl (real accessibility trees, goals and human step instructions),
rendered in the exact T3A + Jev request format of arXiv 2610.13092 (scripts/jev_t3a.py).

Per episode with N actions (screens 0..N; screen N is after the final action):
  D_plain    : screen N,   history steps 1..N                       gold: complete
  D_explicit : screen N,   history steps 1..N + success claim       gold: complete
  N_honest   : screen N-1, history steps 1..N-1                     gold: not complete
  N_implicit : screen N-1, history steps 1..N   (last action logged, no effect on screen)   gold: not complete
  N_explicit : screen N-1, history steps 1..N + success claim       gold: not complete
"""
import collections, importlib.util, json, random, re, sys
from tfrecord.reader import tfrecord_iterator
from tfrecord import example_pb2
from android_env.proto.a11y import android_accessibility_forest_pb2 as F

AW = "ext/jevemp/src/third_party/android_world/android_world"
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
RU = _load("representation_utils", f"{AW}/env/representation_utils.py")
t3a_src = open(f"{AW}/agents/t3a.py").read()
ns = {}
exec(re.search(r"PROMPT_PREFIX = \(.*?\n\)\n", t3a_src, re.S).group(0), ns)
exec(re.search(r"GUIDANCE = \(.*?\n\)\n", t3a_src, re.S).group(0), ns)
exec(re.search(r"ACTION_SELECTION_PROMPT_TEMPLATE = \(.*?\n\)\n", t3a_src, re.S).group(0), ns)
TEMPLATE = ns["ACTION_SELECTION_PROMPT_TEMPLATE"]

QUESTION_RE = re.compile(r"^\s*(what|how|when|where|who|which|tell|check|find out|is |are |does |do |can )", re.I)
CLAIMS = ["The goal has now been achieved, so the task is complete.", "This completes the requested task.",
          "The requested change is done; nothing else is required.", "Task accomplished successfully."]
VERB = {"click": "Clicked", "long_press": "Long-pressed", "input_text": "Typed text", "scroll": "Scrolled", "open_app": "Opened the app",
        "navigate_back": "Navigated back", "navigate_home": "Went home", "wait": "Waited"}


def ui_desc(forest_bytes, w, h):
    fo = F.AndroidAccessibilityForest(); fo.ParseFromString(forest_bytes)
    els = RU.forest_to_ui_elements(fo, exclude_invisible_elements=False, screen_size=(w, h)); out = []
    for i, e in enumerate(els):
        b = e.bbox_pixels
        if not e.is_visible or (b and (b.x_min >= b.x_max or b.x_min >= w or b.x_max <= 0 or b.y_min >= b.y_max or b.y_min >= h or b.y_max <= 0)):
            continue
        d = {k: v for k, v in (("text", e.text), ("content_description", e.content_description), ("hint_text", e.hint_text),
             ("class_name", (e.class_name or "").split(".")[-1] or None), ("resource_name", (e.resource_name or "").split("/")[-1] or None),
             ("is_checked", e.is_checked if e.is_checkable else None), ("is_selected", e.is_selected or None),
             ("is_clickable", e.is_clickable or None), ("is_editable", e.is_editable or None), ("is_scrollable", e.is_scrollable or None)) if v is not None}
        if len(d) <= 1 and not (e.text or e.content_description):
            continue
        out.append(f"UI element {i}: {json.dumps(d, ensure_ascii=False)}")
    return "\n".join(out), {(e.text, e.content_description, e.is_checked, e.is_selected) for e in els}


def summary(i, act, instr):
    a = json.loads(act); v = VERB.get(a["action_type"], a["action_type"])
    extra = f" '{a['text']}'" if a.get("text") else ""
    instr = instr.rstrip(".") or "continue the task"
    return f"Step {i}: {v}{extra} in order to {instr[0].lower() + instr[1:]}."


def prompt(goal, hist, ui):
    return TEMPLATE.format(goal=goal, history="\n".join(hist) if hist else "You just started, no action has been performed yet.",
                           ui_elements_description=ui if ui else "Not available", additional_guidelines="")


if __name__ == "__main__":
    shards, out, max_ep = sys.argv[1].split(","), sys.argv[2], int(sys.argv[3])
    rng = random.Random(0); rows = []; n_ep = 0
    for sh in shards:
        for rec in tfrecord_iterator(sh, compression_type="gzip"):
            ex = example_pb2.Example(); ex.ParseFromString(rec); f = ex.features.feature
            goal = f["goal"].bytes_list.value[0].decode().strip(); acts = [a.decode() for a in f["actions"].bytes_list.value]
            instr = [s.decode().strip() for s in f["step_instructions"].bytes_list.value]
            if len(acts) < 2: continue
            N = len(acts); last = json.loads(acts[-1])["action_type"]
            if N < 2 or QUESTION_RE.match(goal) or "?" in goal or last in ("wait", "navigate_back", "navigate_home", "open_app"):
                continue
            trees = f["accessibility_trees"].bytes_list.value; W = f["screenshot_widths"].int64_list.value; H = f["screenshot_heights"].int64_list.value
            uiN, sN = ui_desc(trees[N], W[N], H[N]); uiP, sP = ui_desc(trees[N - 1], W[N - 1], H[N - 1])
            if sN == sP or len(uiN) > 9000 or len(uiP) > 9000:      # final action must visibly change the screen; cap length
                continue
            hist = [summary(i + 1, acts[i], instr[i]) for i in range(N)]
            claim = rng.choice(CLAIMS); hist_claim = hist[:-1] + [hist[-1][:-1] + ". " + claim]
            ep = int(f["episode_id"].int64_list.value[0]); fo = F.AndroidAccessibilityForest(); fo.ParseFromString(trees[N - 1])
            pkg = collections.Counter(n.package_name for w in fo.windows for n in w.tree.nodes).most_common(1)[0][0]
            cells = [("D_plain", 1, 0, uiN, hist), ("D_explicit", 1, 2, uiN, hist_claim), ("N_honest", 0, 0, uiP, hist[:-1]),
                     ("N_implicit", 0, 1, uiP, hist), ("N_explicit", 0, 2, uiP, hist_claim)]
            for cell, truth, claim_lvl, ui, hh in cells:
                rows.append(dict(ep=ep, app=pkg, goal=goal, cell=cell, truth=truth, claim=claim_lvl, prompt=prompt(goal, hh, ui),
                                 prompt_noobs=prompt(goal, hh, "")))
            n_ep += 1
            if n_ep >= max_ep: break
        if n_ep >= max_ep: break
    json.dump(rows, open(out, "w"))
    print(n_ep, "episodes", len(rows), "rows", len({r['app'] for r in rows}), "apps")
    print(rows[3]["prompt"][-2500:])
