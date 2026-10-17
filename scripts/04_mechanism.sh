#!/usr/bin/env bash
# Activation patching (Section 6, Figure 3, Appendix E) and steering on the synthetic diagnostic set.
source "$(dirname "$0")/common.sh"
python src/mechanism/mechanism.py 100
python src/mechanism/mechanism_gen.py results/alf_eval.json src/common/alf_instr.json 100 results/mechanism_alf.json
python src/mechanism/mechanism_gen.py results/ws_eval.json  src/common/ws_instr.json  100 results/mechanism_ws.json
export MECH_MODEL=Mapika/decider-12b
MECH_OUT=results/mechanism_patching_12b.json python src/mechanism/mechanism.py 100
python src/mechanism/mechanism_gen.py results/alf_eval.json src/common/alf_instr.json 100 results/mechanism_alf_12b.json
python src/mechanism/mechanism_gen.py results/ws_eval.json  src/common/ws_instr.json  100 results/mechanism_ws_12b.json
unset MECH_MODEL
# synthetic diagnostic set and steering
python src/data/gen_data.py 6 results/data.json
python src/scoring/run_pilot.py Mapika/decider-4b results/decider4b
python src/analysis/analyze.py results/decider4b.npz
python src/mechanism/steer.py Mapika/decider-4b results/decider4b.npz 12,16,20,24
