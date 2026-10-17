"""WebShop text environment without Java: a drop-in for pyserini's LuceneSearcher (BM25 via rank_bm25 over the same
document text WebShop indexes: title, description, first bullet point, options; lower-cased, as in
search_engine/convert_product_file_format.py). Everything else is the original web_agent_site code."""
import json, sys, types
from pathlib import Path
from rank_bm25 import BM25Okapi

WS = Path(__file__).resolve().parents[1] / "ext/webshop"
sys.path.insert(0, str(WS))


class _Hit:
    def __init__(self, docid, score): self.docid, self.score = docid, score


class _Doc:
    def __init__(self, raw): self._raw = raw
    def raw(self): return self._raw


class LuceneSearcher:
    def __init__(self, _index_path):
        from web_agent_site.engine.engine import load_products
        from web_agent_site.utils import DEFAULT_FILE_PATH
        prods, *_ = load_products(filepath=DEFAULT_FILE_PATH)
        self.ids, texts = [], []
        for p in prods:
            opts = ", and ".join(f"{k}: {', '.join(v)}" for k, v in p.get("options", {}).items())
            texts.append(" ".join([p["Title"], p["Description"], (p["BulletPoints"] or [""])[0], opts]).lower())
            self.ids.append(p["asin"])
        self.bm25 = BM25Okapi([t.split() for t in texts])

    def search(self, q, k=10):
        s = self.bm25.get_scores(q.lower().split())
        return [_Hit(i, float(s[i])) for i in s.argsort()[::-1][:k]]

    def doc(self, docid): return _Doc(json.dumps({"id": self.ids[docid]}))


mod = types.ModuleType("pyserini.search.lucene"); mod.LuceneSearcher = LuceneSearcher
for name in ("pyserini", "pyserini.search"): sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["pyserini.search.lucene"] = mod


def make_env():
    import gym  # noqa: F401
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    return WebAgentTextEnv(observation_mode="text_rich", num_products=1000, human_goals=False)
