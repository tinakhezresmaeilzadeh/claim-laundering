#!/usr/bin/env bash
# Figures. Needs the hidden states (.npz) from scripts 01-02 and the patching results from 04.
source "$(dirname "$0")/common.sh"
python src/figures/layer_curves.py
python src/figures/make_figs.py figs
