#!/usr/bin/env bash
# Paired states (Section 3).
# Default: unpack the exact states used in the paper (data/paired_states/*.json.gz) into results/. No download, no GPU.
# REBUILD=1 bash scripts/00_build_data.sh rebuilds them from the raw data (needs data/ and ext/ from setup.sh). Phones
# rebuild identically; ALFWorld and WebShop do not, because ALFWorld picks the order of its games and WebShop draws each
# goal's price limit at random, outside our code. The LLM-written summaries need one GPU.
source "$(dirname "$0")/common.sh"
if [ "${REBUILD:-0}" != 1 ]; then
  for f in data/paired_states/*.json.gz; do gunzip -c "$f" > "results/$(basename "$f" .gz)"; done
  echo "unpacked $(ls data/paired_states/*.json.gz | wc -l) paired-state files into results/"; exit 0
fi
A=data/androidcontrol
python src/data/build_ac.py $A/android_control-00000-of-00020,$A/android_control-00001-of-00020 results/ac.json 100000   # 640 episodes
python src/data/make_subset.py results/ac.json data/ac_sub_episodes.json results/ac_sub.json                             # 200 episodes
python src/data/build_alf.py eval_out_of_distribution 134 3 results/alf_eval.json   # unseen games (117 solved in the paper)
python src/data/build_alf.py train 250 3 results/alf_train.json                     # training games (probe fitting, fine-tuning)
python src/data/build_ws.py 0 300 results/ws_eval.json                              # test goals
python src/data/build_ws.py 3000 300 results/ws_train.json                          # training goals
python src/data/llm_summaries.py Qwen/Qwen3.5-9B results/ac_llm.json                # LLM-written step summaries (T3A prompt)
python src/data/gen_data.py 6 results/data.json                                     # synthetic diagnostic set
