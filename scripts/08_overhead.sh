#!/usr/bin/env bash
# Serving latency (Appendix). bench_kev.py runs in the Kev environment (venv_kev).
source "$(dirname "$0")/common.sh"
python src/bench_overhead.py Mapika/decider-4b 30
python src/bench_engine.py 30
echo "Then, in venv_kev: python src/bench_kev.py 20"
