#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# The book runs used Python 3.10.4.
PYTHON="${PYTHON:-python3.10}"
if [[ ! -d .venv ]]; then
  "${PYTHON}" -m venv .venv
fi
source .venv/bin/activate
python -m pip install pip==22.0.4
python -m pip install -r requirements.txt
# PET's setup.py only lists top-level packages, so it works as an editable
# install only; --no-deps because requirements.txt pins its dependencies.
python -m pip install --no-deps --src .venv/src \
  -e "git+https://github.com/KriFos1/PET.git@d478e325dc1a146a25703b44319605f2820130a8#egg=PET"

python scripts/build_levels.py
python scripts/make_prior.py
python scripts/run_experiment.py "${EXPERIMENT:-ES-MDA}"

printf '\nSetup and %s run complete. Activate with: source .venv/bin/activate\n' "${EXPERIMENT:-ES-MDA}"
printf 'Run all experiments with: ./run_all.sh\n'
