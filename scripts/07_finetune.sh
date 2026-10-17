#!/usr/bin/env bash
# Fine-tuning decider-4b on paired states (Section 9, Table 6).
source "$(dirname "$0")/common.sh"
python src/finetune/cit_eval.py base results/cit_eval_base.json
python src/finetune/cit_train.py results/cit_adapter                                   # phones only (2,755 states)
python src/finetune/cit_eval.py results/cit_adapter results/cit_eval_tuned.json
python src/finetune/cit_train.py results/cit_adapter_all 1.0 phones,alfworld,webshop 1000    # 1,000 states per domain
python src/finetune/cit_eval.py results/cit_adapter_all results/cit_eval_all.json
python src/finetune/cit_train.py results/cit_adapter_all_fullphones 1.0 phones:all,alfworld:1000,webshop:1000
python src/finetune/cit_eval.py results/cit_adapter_all_fullphones results/cit_eval_all_fullphones.json
