#!/usr/bin/env bash
# Live agents (Table 5). Order matters: 'collect' records training-task states for EviGate; 'verify' and 'collect'
# write the veto rates that the random controls are matched to.
source "$(dirname "$0")/common.sh"
for g in collect none verify probe probe2 random2; do python src/agents/alf_e2e.py $g 0 results/alf_e2e_$g.json 134; done
for g in collect none verify probe probe2 random2; do python src/agents/ws_e2e.py  $g 0 results/ws_e2e_$g.json 200; done
# phone simulator: 24 tasks x 30 seeds, two folds (EviGate fit on one half of the tasks, tested on the other)
for f in 0 1; do python src/agents/sim_e2e_v2.py 30 $f none,verify,probe; python src/agents/sim_e2e_v2.py 30 $f probe,random4; done
