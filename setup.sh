#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
PYTHON="${PYTHON:-python3.12}"

if [[ ! -d .venv ]]; then
  "${PYTHON}" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

python scripts/check_archived_levels.py
python scripts/make_prior.py
python scripts/run_experiment.py "${EXPERIMENT:-ES-MDA}"

printf '\nSetup and %s run complete. Activate with: source .venv/bin/activate\n' "${EXPERIMENT:-ES-MDA}"
printf 'Run all experiments with: ./run_all.sh\n'
