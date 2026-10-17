# Claim Laundering: When Agent Decision Models Trust Records Over Reality

Code for the paper by **Tina Khezresmaeilzadeh, Seyedarmin Azizi, Mehdi Kamal and Massoud Pedram**
(University of Southern California).

Agents keep a record of what they did, and fast decision models decide from that record. When the record says an action
worked but the world shows it did not, the decision model follows the record. This repository contains everything needed
to reproduce the paper:

- **Paired states** that differ in one factor (phones, ALFWorld, WebShop), and the effect of each factor on the decision.
- **Probes** showing that the hidden states still encode whether the action worked.
- **Activation patching** locating where the record's influence takes over (decider-4b and decider-12b).
- **EviGate**, a linear probe on a middle layer that vetoes terminal decisions, tested offline and in live agents
  against random vetoes at the same rate.
- **Fine-tuning** the decision model on paired states, as the alternative to gating.

## The five paired states

Each decision point is rendered five times. Only one factor changes between neighbouring states:

| State | The last action... | Record | Correct decision |
|---|---|---|---|
| Done | worked | logged | terminal (stop / proceed / buy) |
| Done+claim | worked | logged + success claim | terminal |
| Honest | not attempted yet | not logged | not terminal |
| Logged | failed silently | logged | not terminal |
| Logged+claim | failed silently | logged + success claim | not terminal |

In the code these are `D_plain`, `D_explicit`, `N_honest`, `N_implicit` and `N_explicit` (ALFWorld/WebShop builders use
`S_plain`, `S_think`, `F_honest`, `F_implicit`, `F_explicit`).

## Setup

Tested with Python 3.12 on two RTX A6000 (48 GB) GPUs.

```bash
./setup.sh
```

This creates two environments (`venv` for everything, `venv_kev` for the Kev models, which pin an older torch), clones
[decider](https://github.com/Mapika/decider), [Kev](https://github.com/jaredpalmer/kev),
[Laya](https://github.com/NandhaKishorM/laya) and [WebShop](https://github.com/princeton-nlp/WebShop) at the commits we
used, and downloads AndroidControl shards 0 and 1. Two manual steps remain:

1. **Code of arXiv 2610.13092** (the Jev agent adapters: T3A prompts, AndroidWorld UI conversion, the ALFWorld ReAct-J
   adapter). Place it in `ext/jevemp/src` and run `pip install -e ext/jevemp/src/third_party/alfworld`.
2. **WebShop data**: the 1,000-product set (`cd ext/webshop && ./setup.sh -d small`). ALFWorld data is downloaded by
   `setup.sh` into `data/alfworld`.

Models are loaded from the Hugging Face Hub: `Mapika/decider-{2b,4b,12b}`, `jaredpalmer/kev-{0.8b,4b,9b}`,
`convaiinnovations/laya`, `Qwen/Qwen3.5-4B(-Base)` and `Qwen/Qwen3.5-9B` (LLM summaries, agent thoughts).
Hosted Jev is called through OpenRouter and needs a key in `~/.openrouter_key` (`chmod 600`).

## Reproducing the paper

Run the scripts in order from the repository root with `venv` active. Each script states what it reproduces.

| Script | Paper |
|---|---|
| `scripts/00_build_data.sh` | Section 3: paired states, LLM-written summaries |
| `scripts/01_phones.sh` | Tables 2-4 (phones), seven models, backbone comparison |
| `scripts/02_alfworld_webshop.sh` | Tables 2-4 (ALFWorld, WebShop) |
| `scripts/03_hosted_jev.sh` | Hosted Jev rows; paired CIs for Table 2 (Appendix B) |
| `scripts/04_mechanism.sh` | Section 6, Figure 3, decider-12b patching; steering |
| `scripts/05_live_agents.sh` | Table 5 (live agents) |
| `scripts/06_remove_record.sh` | Appendix: removing the record |
| `scripts/07_finetune.sh` | Section 9, Table 6 |
| `scripts/08_overhead.sh` | Appendix: serving latency |
| `scripts/09_best_of_n.sh` | Appendix: the probe as a best-of-N verifier |
| `scripts/10_figures.sh` | Figures |

All outputs go to `results/`. The hidden states (`.npz`) are large and are not in the repository.

**Live agents (Table 5).** Each agent runs once per task with 30% injected silent failures (ALFWorld, WebShop). The phone
simulator runs 24 tasks x 30 seeds with silent failures on 40% of first attempts and 20% of retries, and its summarizer
writes a success claim after 70% of failed taps. Gate names in the code:

| Paper | `alf_e2e.py` / `ws_e2e.py` | `sim_e2e_v2.py` |
|---|---|---|
| No gate | `none` | `none` |
| Second question | `verify` | `verify` |
| EviGate, fit on offline contrast sets | `probe` | — |
| EviGate, fit on target-domain states | `probe2` (states from `collect` on training tasks) | `probe` (other half of the tasks) |
| Random veto, matched rate | `random2` | `random4` |

`collect` must run before `probe2` and `random2`; the order in `scripts/05_live_agents.sh` handles this.

## Results included

`results/summaries/` holds the analysis outputs behind every table and figure (effects, probe AUROCs, gate results,
patching curves, live-agent summaries, fine-tuning evaluations, latency), so the numbers can be checked without a GPU.
`data/ac_sub_episodes.json` lists the 200 AndroidControl episodes of the phone subset used for the larger models,
hosted Jev, the LLM summaries and patching.

## Code map

| File | What it does |
|---|---|
| `build_ac.py`, `make_subset.py`, `llm_summaries.py` | Phone paired states from AndroidControl in the T3A/Jev format; LLM-written summaries |
| `build_alf.py`, `alf_common.py` | ALFWorld paired states from expert trajectories |
| `build_ws.py`, `ws_common.py` | WebShop paired states; WebShop search without Java (BM25) |
| `run_ac.py`, `run_alf.py`, `run_jev.py` | Score states with open models (probabilities + hidden states) or hosted Jev |
| `analyze_ac.py`, `analyze_final.py`, `analyze_alf.py`, `analyze_jev.py`, `analyze_llm.py`, `table2_ci.py` | Effects, probes (GroupKFold by app/game/goal, layer chosen inside training folds), offline gates, CIs |
| `mechanism.py`, `mechanism_gen.py` | Activation patching (phones; ALFWorld/WebShop); `MECH_MODEL` selects the model |
| `gen_data.py`, `run_pilot.py`, `analyze.py`, `steer.py` | Synthetic diagnostic set and activation steering |
| `alf_e2e.py`, `ws_e2e.py`, `sim_e2e_v2.py`, `analyze_e2e.py`, `merge_ws.py` | Live agents with gates |
| `story_blind.py`, `analyze_story_blind.py` | Removing the record from layer *l* onward (`SB_LAYERS=0`: from the input) |
| `cit_train.py`, `cit_eval.py` | LoRA fine-tuning on paired states and its evaluation |
| `bench_overhead.py`, `bench_engine.py`, `bench_kev.py` | Serving latency of EviGate |
| `ws_bon.py`, `analyze_bon.py` | Best-of-N with the probe as verifier |
| `layer_curves.py`, `make_figs.py` | Figures |

## Citation

```bibtex
@article{khezresmaeilzadeh2026claim,
  title   = {Claim Laundering: When Agent Decision Models Trust Records Over Reality},
  author  = {Khezresmaeilzadeh, Tina and Azizi, Seyedarmin and Kamal, Mehdi and Pedram, Massoud},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). AndroidControl, ALFWorld, WebShop and the models keep their own licenses.

Contact: Tina Khezresmaeilzadeh, khezresm@usc.edu
