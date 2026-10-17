#!/usr/bin/env bash
# Phones: effects (Table 2), probes (Table 3), offline gates (Table 4), all seven models and the backbone (Appendix).
source "$(dirname "$0")/common.sh"
run() { local py=python; [ "$1" = kev ] && py=$PY_KEV
        $py src/scoring/run_ac.py "$1" "$2" "$3" "$4" && python src/analysis/analyze_ac.py "$3" "$4" "$5"; }
run decider Mapika/decider-4b      results/ac.json     results/ac_decider4b.npz     decider-4b
python src/analysis/analyze_final.py results/ac.json results/ac_decider4b.npz decider-4b
run kev     jaredpalmer/kev-4b     results/ac.json     results/ac_kev4b_full.npz    Kev-4B
python src/analysis/analyze_final.py results/ac.json results/ac_kev4b_full.npz Kev-4B
run decider Mapika/decider-4b      results/ac_sub.json results/ac_decider4b_sub.npz decider-4b
run decider Mapika/decider-2b      results/ac_sub.json results/ac_decider2b.npz     decider-2b
run decider Mapika/decider-12b     results/ac_sub.json results/ac_decider12b.npz    decider-12b
run kev     jaredpalmer/kev-0.8b   results/ac_sub.json results/ac_kev08b.npz        Kev-0.8B
run kev     jaredpalmer/kev-4b     results/ac_sub.json results/ac_kev4b.npz         Kev-4B
run kev     jaredpalmer/kev-9b     results/ac_sub.json results/ac_kev9b.npz         Kev-9B
run laya    convaiinnovations/laya results/ac_sub.json results/ac_laya.npz          Laya
run decider Qwen/Qwen3.5-4B-Base   results/ac_sub.json results/ac_qwenbase4b.npz    Qwen3.5-4B-Base   # backbone comparison
run decider Qwen/Qwen3.5-4B        results/ac_sub.json results/ac_qwenchat4b.npz    Qwen3.5-4B
for n in decider4b_sub decider2b decider12b kev08b kev4b kev9b qwenbase4b qwenchat4b; do
  python src/analysis/analyze_final.py results/ac_sub.json results/ac_$n.npz $n; done
# LLM-written summaries
python src/scoring/run_ac.py decider Mapika/decider-4b results/ac_llm.json results/ac_llm_decider4b.npz
python src/analysis/analyze_llm.py results/ac_llm.json results/ac_llm_decider4b.npz results/ac.json results/ac_decider4b.npz
