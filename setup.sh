#!/usr/bin/env bash
# Environments, external code and data. Run once from the repository root.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ext data/androidcontrol results figs

# 1. main environment
python3.12 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/Mapika/decider ext/decider && git -C ext/decider checkout d6fc522 && pip install -e ext/decider
git clone https://github.com/NandhaKishorM/laya ext/laya && git -C ext/laya checkout c73908d && pip install -e ext/laya
git clone https://github.com/princeton-nlp/WebShop ext/webshop && git -C ext/webshop checkout 64fa2a5
python -m spacy download en_core_web_sm
deactivate

# 2. Kev environment (Kev pins an older torch); only for Kev models and bench_kev.py
python3.12 -m venv venv_kev && . venv_kev/bin/activate
pip install -r requirements_kev.txt
git clone https://github.com/jaredpalmer/kev ext/kev && git -C ext/kev checkout 85ccf56 && pip install -e ext/kev
deactivate

# 3. Code of arXiv 2610.13092 (T3A prompts, AndroidWorld UI conversion, ALFWorld ReAct-J adapter), placed in ext/jevemp/src.
#    It is distributed by its authors (see the paper); after placing it:
#      . venv/bin/activate && pip install -e ext/jevemp/src/third_party/alfworld
[ -d ext/jevemp/src ] || echo "ACTION NEEDED: put the code of arXiv 2610.13092 in ext/jevemp/src (see README)"

# 4. data
for s in 00000 00001; do
  f=data/androidcontrol/android_control-$s-of-00020
  [ -f $f ] || curl -L -o $f https://storage.googleapis.com/gresearch/android_control/android_control-$s-of-00020
done
curl -L -o data/androidcontrol/splits.json https://storage.googleapis.com/gresearch/android_control/splits.json
mkdir -p data/alfworld
for z in 0.2.2/json_2.1.1_json.zip 0.4.2/json_2.1.3_tw-pddl.zip; do
  f=data/alfworld/$(basename $z); [ -f $f ] || curl -L -o $f https://github.com/alfworld/alfworld/releases/download/$z
  unzip -qn $f -d data/alfworld
done
echo "WebShop data (1,000-product set): cd ext/webshop && ./setup.sh -d small"
