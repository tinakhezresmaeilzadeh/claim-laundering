#!/usr/bin/env bash
# Paired states (Section 3). Needs data/ and ext/ from setup.sh. CPU only, except the LLM summaries (one GPU).
source "$(dirname "$0")/common.sh"
A=data/androidcontrol
python src/build_ac.py $A/android_control-00000-of-00020,$A/android_control-00001-of-00020 results/ac.json 100000   # 640 episodes
python src/make_subset.py results/ac.json data/ac_sub_episodes.json results/ac_sub.json                             # 200 episodes
python src/build_alf.py eval_out_of_distribution 134 3 results/alf_eval.json   # 117 solved unseen games
python src/build_alf.py train 250 3 results/alf_train.json                     # training games (probe fitting, fine-tuning)
python src/build_ws.py 0 300 results/ws_eval.json                              # test goals
python src/build_ws.py 3000 300 results/ws_train.json                          # training goals
python src/llm_summaries.py Qwen/Qwen3.5-9B results/ac_llm.json                # LLM-written step summaries (T3A prompt)
