#!/usr/bin/env bash
# ALFWorld and WebShop: effects, probes and offline gates (Tables 2-4).
source "$(dirname "$0")/common.sh"
for d in alf ws; do
  I=src/alf_instr.json; [ $d = ws ] && I=src/ws_instr.json
  python src/run_alf.py decider Mapika/decider-4b  results/${d}_eval.json  results/${d}_eval_decider4b.npz  $I
  python src/run_alf.py decider Mapika/decider-12b results/${d}_eval.json  results/${d}_eval_decider12b.npz $I
  python src/run_alf.py kev     jaredpalmer/kev-4b results/${d}_eval.json  results/${d}_eval_kev4b.npz     $I
  python src/run_alf.py decider Mapika/decider-4b  results/${d}_train.json results/${d}_train_decider4b.npz $I
  for n in decider4b decider12b kev4b; do python src/analyze_alf.py results/${d}_eval.json results/${d}_eval_$n.npz $n; done
done
