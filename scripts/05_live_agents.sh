#!/usr/bin/env bash
# Live agents (Table 5). Order matters: 'collect' records training-task states for EviGate; 'verify' and 'collect'
# write the veto rates that the random controls are matched to.
source "$(dirname "$0")/common.sh"
# ALFWorld: collect on 100 training-split games (148 gate states); test on the 134 unseen games
python src/agents/alf_e2e.py collect 0 results/alf_e2e_collect.json 100
for g in none verify probe probe2 random2; do python src/agents/alf_e2e.py $g 0 results/alf_e2e_$g.json 134; done
# WebShop: collect on goals 1000-1299 (480 gate states); test on goals 0-199. random2 uses probe2's veto rate, so it runs last
python src/agents/ws_e2e.py collect 0 results/ws_e2e_collect.json 300 1000
for g in none verify probe probe2 random2; do python src/agents/ws_e2e.py $g 0 results/ws_e2e_$g.json 200; done
# Phone simulator: 24 tasks x 30 seeds, two folds (EviGate fit on one half of the tasks, tested on the other)
for f in 0 1; do
  python src/agents/sim_e2e.py 30 $f                                  # none, verify, contrast, probe, random (verify's rate)
  python src/agents/sim_e2e_v2.py 30 $f none,verify,probe,llmjudge   # adds the 9B LLM judge (Appendix G)
  python src/agents/sim_e2e_v2.py 30 $f probe,random4                # random veto at EviGate's own rate (Table 5)
done
python src/agents/analyze_e2e.py 30                                   # pooled success and paired CIs over tasks
