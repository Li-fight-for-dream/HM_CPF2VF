#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

log_dir="logs"
mkdir -p "${log_dir}"

ts="$(date +%Y%m%d_%H%M%S)"
log_file="${log_dir}/recipe_PCANet_${ts}.log"

nohup python -m src.PCANet_trainer.train --config configs/recipe_PCANet.yaml >"${log_file}" 2>&1 &
pid=$!

echo "Started PCANet training in background."
echo "PID: ${pid}"
echo "Log: ${log_file}"
