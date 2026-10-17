# Claim Laundering: When Agent Decision Models Trust Records Over Reality

**Code and reproduction instructions for the paper** by Tina Khezresmaeilzadeh, Seyedarmin Azizi, Mehdi Kamal, and
Massoud Pedram (University of Southern California).

Agent decision models choose actions from inputs that contain both the current observation of the environment and the
agent's record of its earlier actions. These sources can disagree: an action may fail silently while the record says it
succeeded. We study whether decision models follow the observed state or the agent's record when making consequential
decisions such as **stop**, **proceed**, or **buy**.

This repository provides code for:

- **Controlled counterfactual evaluations:** five paired state variants in Android phone tasks, ALFWorld, and WebShop
  that isolate the influence of observations, logged actions, and success claims.
- **Hidden-state probing:** linear probes that test whether action success is recoverable from internal representations
  even when the model's final decision is wrong.
- **Activation patching:** layer-wise interventions measuring when record-dependent information influences the decision
  (decider-4b and decider-12b).
- **EviGate:** an intermediate-layer linear readout that vetoes unsupported commitment decisions without changing the
  decision model's weights.
- **Fine-tuning:** LoRA adaptation on paired states, evaluated as an alternative to gating.

The repository includes experiment scripts and precomputed summary results. Full reproduction also requires external
datasets and agent adapters, model downloads, GPUs, and an OpenRouter key for the hosted Jev experiments.

## 1. Experimental design: five state variants

For each decision point, we construct five variants:

| State | Outcome of the last action | Agent record | Expected decision |
| --- | --- | --- | --- |
| **Done** | Succeeded | Attempt logged | Commit |
| **Done+claim** | Succeeded | Attempt logged; success claimed | Commit |
| **Honest** | Not yet attempted | No attempt logged | Do not commit |
| **Logged** | Attempted, no effect | Attempt logged | Do not commit |
| **Logged+claim** | Attempted, no effect | Attempt logged; success falsely claimed | Do not commit |

*Commit* means `complete` on phones, proceeding to the next step in ALFWorld (instead of retrying the last action), and
`buy now` in WebShop. In ALFWorld, proceeding does not end the episode.

The controlled comparisons are:

| Comparison | What changes |
| --- | --- |
| **Done vs. Logged** | The observation, with the action log held fixed |
| **Logged vs. Honest** | The logged attempt, with the observation held fixed |
| **Logged+claim vs. Logged** | The explicit success claim, with the failed observation held fixed |
| **Done+claim vs. Logged+claim** | The actual outcome, with a success claim present in both |

In the code, phone states are named `D_plain`, `D_explicit`, `N_honest`, `N_implicit`, and `N_explicit`; the ALFWorld and
WebShop builders use `S_plain`, `S_think`, `F_honest`, `F_implicit`, and `F_explicit`.

## 2. Installation

Experiments were run with **Python 3.12** on **two NVIDIA RTX A6000 GPUs (48 GB each)**. From the repository root:

```bash
./setup.sh
source venv/bin/activate
```

The setup script creates two environments:

- `venv` for the main experiments;
- `venv_kev` for the Kev models, which pin an older PyTorch. The scripts call it automatically for Kev runs.

It also checks out the pinned versions of [decider](https://github.com/Mapika/decider),
[Kev](https://github.com/jaredpalmer/kev), [Laya](https://github.com/NandhaKishorM/laya), and
[WebShop](https://github.com/princeton-nlp/WebShop), and downloads AndroidControl shards 0 and 1 and the ALFWorld data.

### Additional setup

**Jev agent adapters.** Obtain the code accompanying arXiv:2610.13092 (T3A prompts, AndroidWorld UI conversion, and the
ALFWorld ReAct-J adapter), place it at `ext/jevemp/src`, and install its ALFWorld component:

```bash
pip install -e ext/jevemp/src/third_party/alfworld
```

**WebShop data.** Install the 1,000-product set:

```bash
cd ext/webshop && ./setup.sh -d small && cd ../..
```

### Models and credentials

Models are downloaded from the Hugging Face Hub when first used:

- decider: `Mapika/decider-{2b,4b,12b}`
- Kev: `jaredpalmer/kev-{0.8b,4b,9b}`
- Laya: `convaiinnovations/laya`
- Qwen: `Qwen/Qwen3.5-4B-Base` and `Qwen/Qwen3.5-4B` (backbone comparison; `Qwen3.5-4B` also writes the WebShop agent's
  searches and thoughts) and `Qwen/Qwen3.5-9B` (LLM-written summaries, the ALFWorld agent's thoughts)

Hosted Jev is called through OpenRouter. Put your key in `~/.openrouter_key` and restrict access to it
(`chmod 600 ~/.openrouter_key`). Never commit API keys.

## 3. Reproducing the experiments

Run the scripts **from the repository root**, in this order, with `venv` active:

| Script | Reproduces |
| --- | --- |
| `scripts/00_build_data.sh` | Section 3: paired states and LLM-written summaries |
| `scripts/01_phones.sh` | Phone results in Tables 2–4; seven open models and the backbone comparison |
| `scripts/02_alfworld_webshop.sh` | ALFWorld and WebShop results in Tables 2–4 |
| `scripts/03_hosted_jev.sh` | Hosted Jev results; paired confidence intervals for Table 2 |
| `scripts/04_mechanism.sh` | Section 6: activation patching (Figure 2; decider-12b in Table 14) and steering |
| `scripts/05_live_agents.sh` | Table 5: live agents and matched random vetoes |
| `scripts/06_remove_record.sh` | Appendix: removing the record from later layers and from the input |
| `scripts/07_finetune.sh` | Section 9 and Table 6: fine-tuning |
| `scripts/08_overhead.sh` | Appendix: serving latency |
| `scripts/09_best_of_n.sh` | Appendix: the probe as a best-of-*N* verifier |
| `scripts/10_figures.sh` | Figures 2, 4 and 5 |

For example:

```bash
bash scripts/00_build_data.sh
bash scripts/01_phones.sh
```

Outputs are written to `results/`. The hidden-state files (`.npz`) are large and not included; the scripts regenerate them.

### Live-agent protocol (Table 5)

**ALFWorld and WebShop.** Agents run once per test task, with a **30% probability that an effectful action fails
silently**. EviGate is fit on states recorded from the same agent on separate training tasks. We compare no gate, a
second verification question, EviGate, and random vetoes. In WebShop the random veto uses EviGate's measured veto rate;
in ALFWorld it uses the true failure rate among gated states on the training tasks (79.7%; the measured random veto
rate was 78.6%, vs. 75.5% for EviGate).

**Phone simulator.** A text simulator, **not Android emulation**: 24 tasks across 11 apps, 30 seeds per task. The
decisive tap fails with probability **40% on the first attempt** and **20% on retries**. After a failed tap, a
**stochastic record generator** (not an LLM summarizer) writes a success claim with probability **70%**. EviGate is fit on
simulator states from the other half of the tasks, and the random veto uses EviGate's measured rate.

Implementation names of the Table 5 conditions:

| Paper condition | `alf_e2e.py` / `ws_e2e.py` | `sim_e2e_v2.py` |
| --- | --- | --- |
| No gate | `none` | `none` |
| Second question | `verify` | `verify` |
| EviGate, fit on offline contrast sets | `probe` | not run |
| EviGate, fit on target-domain states | `probe2` (states from `collect`) | `probe` (states from the other half of the tasks) |
| Random veto, matched rate | `random2` | `random4` |

`collect` must run before `probe2` and `random2`; `scripts/05_live_agents.sh` runs them in the right order.

## 4. Precomputed results

`results/summaries/` contains the analysis outputs behind the tables and figures:

- paired input effects and confidence intervals;
- probe AUROCs and offline gate results;
- activation-patching curves;
- live-agent results and random-veto controls;
- fine-tuning evaluations;
- serving-latency measurements.

They let readers check the reported numbers without rerunning the GPU experiments.

`data/ac_sub_episodes.json` lists the 200 AndroidControl episodes of the phone subset. This subset is used for the larger
models, hosted Jev, the LLM-written summaries, and activation patching.

## 5. Code layout

| Folder | Files | Purpose |
| --- | --- | --- |
| `src/common/` | `alf_common.py`, `ws_common.py`, `bench_overhead_common.py`, `alf_instr.json`, `ws_instr.json` | Shared environment helpers, decision questions, and instructions |
| `src/data/` | `build_ac.py`, `make_subset.py`, `llm_summaries.py`, `build_alf.py`, `build_ws.py`, `gen_data.py` | Build the paired states (phones, ALFWorld, WebShop, synthetic set); WebShop search runs without Java (BM25) |
| `src/scoring/` | `run_ac.py`, `run_alf.py`, `run_jev.py`, `run_pilot.py` | Score states with open models (probabilities and hidden states) or hosted Jev (probabilities) |
| `src/analysis/` | `analyze_ac.py`, `analyze_final.py`, `analyze_alf.py`, `analyze_jev.py`, `analyze_llm.py`, `analyze.py`, `table2_ci.py` | Paired effects, probes, offline gates, confidence intervals |
| `src/mechanism/` | `mechanism.py`, `mechanism_gen.py`, `steer.py`, `story_blind.py`, `analyze_story_blind.py` | Activation patching (`MECH_MODEL` selects the model), steering, removing the record |
| `src/agents/` | `alf_e2e.py`, `ws_e2e.py`, `sim_e2e_v2.py`, `sim_e2e.py`, `analyze_e2e.py`, `merge_ws.py`, `ws_bon.py`, `analyze_bon.py` | Live agents with gates; best-of-*N* |
| `src/finetune/` | `cit_train.py`, `cit_eval.py` | LoRA fine-tuning on paired states and its evaluation |
| `src/bench/` | `bench_overhead.py`, `bench_engine.py`, `bench_kev.py` | Serving overhead of EviGate |
| `src/figures/` | `layer_curves.py`, `make_figs.py` | Figures and per-layer probe curves |

Probes are evaluated with groups held out by app (phones), game (ALFWorld), or goal (WebShop), and the probe layer is
chosen inside the training folds.

## 6. Citation

```bibtex
@misc{khezresmaeilzadeh2026claim,
  title  = {Claim Laundering: When Agent Decision Models Trust Records Over Reality},
  author = {Khezresmaeilzadeh, Tina and Azizi, Seyedarmin and Kamal, Mehdi and Pedram, Massoud},
  year   = {2026},
  note   = {Preprint}
}
```

## 7. License and contact

Released under the **MIT License** (see `LICENSE`). AndroidControl, ALFWorld, WebShop, the model checkpoints, and other
third-party dependencies keep their own licenses.

**Contact:** Tina Khezresmaeilzadeh, [khezresm@usc.edu](mailto:khezresm@usc.edu)
