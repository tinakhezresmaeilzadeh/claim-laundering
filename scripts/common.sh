# Sourced by every script: run from the repository root, with the main environment active.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p results figs
# Kev models need their own environment (older torch); scripts call it directly for Kev runs.
PY_KEV=${PY_KEV:-venv_kev/bin/python}
