#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/.venv/bin/python"

for experiment in ES-MDA ES-MDA-loc MLHES-MDA SMLES; do
  result="${ROOT}/${experiment}/Results/posterior_state_estimate.npz"
  if [[ -f "${result}" ]]; then
    printf 'Skipping %s: posterior artifact already exists\n' "${experiment}"
  else
    "${PYTHON}" "${ROOT}/scripts/run_experiment.py" "${experiment}"
  fi
done

"${PYTHON}" "${ROOT}/scripts/make_plots.py"
