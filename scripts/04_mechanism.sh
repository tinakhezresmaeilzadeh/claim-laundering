#!/usr/bin/env bash
# Activation patching (Section 6, Figure 3, Appendix E) and steering on the synthetic diagnostic set.
source "$(dirname "$0")/common.sh"
python src/mechanism.py 100
python src/mechanism_gen.py results/alf_eval.json src/alf_instr.json 100 results/mechanism_alf.json
python src/mechanism_gen.py results/ws_eval.json  src/ws_instr.json  100 results/mechanism_ws.json
export MECH_MODEL=Mapika/decider-12b
MECH_OUT=results/mechanism_patching_12b.json python src/mechanism.py 100
python src/mechanism_gen.py results/alf_eval.json src/alf_instr.json 100 results/mechanism_alf_12b.json
python src/mechanism_gen.py results/ws_eval.json  src/ws_instr.json  100 results/mechanism_ws_12b.json
unset MECH_MODEL
# synthetic diagnostic set and steering
python src/gen_data.py 6 results/data.json
python src/run_pilot.py Mapika/decider-4b results/decider4b
python src/analyze.py results/decider4b.npz
python src/steer.py Mapika/decider-4b results/decider4b.npz 12,16,20,24
