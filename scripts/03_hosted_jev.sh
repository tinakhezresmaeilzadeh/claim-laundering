#!/usr/bin/env bash
# Hosted Jev (behaviour only). Needs an OpenRouter key in ~/.openrouter_key (chmod 600).
source "$(dirname "$0")/common.sh"
python src/run_jev.py ac  results/ac_sub.json   results/jev_ac  && python src/analyze_jev.py ac  results/ac_sub.json   results/jev_ac.jsonl
python src/run_jev.py alf results/alf_eval.json results/jev_alf && python src/analyze_jev.py alf results/alf_eval.json results/jev_alf.jsonl
python src/run_jev.py ws  results/ws_eval.json  results/jev_ws  && python src/analyze_jev.py ws  results/ws_eval.json  results/jev_ws.jsonl
python src/table2_ci.py        # paired CIs for all 12 model x domain configurations (Appendix B)
