#!/usr/bin/env bash
# The probe as a best-of-N verifier in WebShop (Appendix).
source "$(dirname "$0")/common.sh"
python src/ws_bon.py 16 results/ws_bon_all.json 0 50
python src/analyze_bon.py results/ws_bon_all.json
