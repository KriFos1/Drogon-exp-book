#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/.venv/bin/python"

for experiment in ES-MDA ES-MDA-loc MLHES-MDA SMLES; do
  if compgen -G "${ROOT}/${experiment}/posterior_forecast.*" > /dev/null; then
    printf 'Skipping %s: posterior forecast already exists\n' "${experiment}"
  else
    "${PYTHON}" "${ROOT}/scripts/run_experiment.py" "${experiment}"
  fi
done

"${PYTHON}" "${ROOT}/scripts/make_figures.py"
