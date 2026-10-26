#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python3.12}"
if [[ ! -d .venv ]]; then
  "${PYTHON}" -m venv .venv
fi
source .venv/bin/activate
# The exact environment that reproduces the book runs.
python -m pip install pip==26.2.1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

# The priors need the numpy of the book runs (see scripts/make_prior.py).
if [[ ! -d .venv-prior ]]; then
  "${PYTHON}" -m venv .venv-prior
fi
.venv-prior/bin/python -m pip install pip==26.2.1
.venv-prior/bin/python -m pip install -r requirements-prior.txt

python scripts/build_levels.py
.venv-prior/bin/python scripts/make_prior.py
python scripts/run_experiment.py "${EXPERIMENT:-ES-MDA}"

printf '\nSetup and %s run complete. Activate with: source .venv/bin/activate\n' "${EXPERIMENT:-ES-MDA}"
printf 'Run all experiments with: ./run_all.sh\n'
