#!/usr/bin/env bash
# Removing the record (Appendix): from layer l onward, and from the input.
source "$(dirname "$0")/common.sh"
for d in phones alfworld webshop; do python src/mechanism/story_blind.py $d results/sb_$d.json; done
python src/mechanism/analyze_story_blind.py results/sb_phones.json results/sb_alfworld.json results/sb_webshop.json
for d in phones alfworld webshop; do SB_LAYERS=0 python src/mechanism/story_blind.py $d results/rr_$d.json; done
python src/mechanism/analyze_story_blind.py results/rr_phones.json results/rr_alfworld.json results/rr_webshop.json
