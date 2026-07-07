#!/usr/bin/env bash
# Shared helpers for pipeline_short / pipeline_medium / pipeline_long.

set -euo pipefail

pipeline_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    export PYTHONPATH=.
    export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
}

pipeline_banner() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# Exit 0 when data/pusht.lance exists and the current user can read Lance files.
pipeline_pusht_lance_readable() {
    python3 - <<'PY'
import os
import sys
from pathlib import Path

path = Path("data/pusht.lance")
if not path.is_dir():
    sys.exit(1)

for sub in ("_transactions", "_versions", "data"):
    subdir = path / sub
    if not subdir.is_dir():
        sys.exit(1)
    sample = next((p for p in subdir.iterdir() if p.is_file()), None)
    if sample is None or not os.access(sample, os.R_OK):
        sys.exit(1)

sys.exit(0)
PY
}

# Collect PushT Lance data when missing, unreadable, or FORCE_COLLECT=1.
pipeline_ensure_pusht_data() {
    local episodes="$1"
    local processes="$2"
    local need_collect=0

    if [[ ! -d data/pusht.lance ]] || [[ "${FORCE_COLLECT:-0}" == "1" ]]; then
        need_collect=1
    elif ! pipeline_pusht_lance_readable; then
        echo "[warn] data/pusht.lance exists but is not readable (often root-owned files); recollecting"
        need_collect=1
    fi

    if [[ "${need_collect}" == "1" ]]; then
        pipeline_banner "Collect data (${episodes} episodes)"
        # PushT collection uses WeakPolicy by default (see collect_data.py).
        # Set FORCE_COLLECT=1 to replace old random-policy data.
        python3 scripts/collect_data.py \
            --episodes "${episodes}" \
            --out data/pusht.lance \
            --processes "${processes}" \
            --num-envs 2 \
            --overwrite
    else
        echo "[skip] data/pusht.lance exists and is readable (set FORCE_COLLECT=1 to recollect)"
    fi
}

pipeline_train() {
    local model="$1"
    local steps="$2"
    local batch="$3"
    local workers="$4"
    local max_episodes="${5:-}"
    local out_dir="outputs/pusht_${model}_seed0"

    pipeline_banner "Train ${model} (${steps} steps)"
    local extra=()
    if [[ -n "${max_episodes}" ]]; then
        extra+=("data.max_episodes=${max_episodes}")
    fi
    python3 scripts/train.py \
        "model=${model}" \
        data=pusht \
        "train.steps=${steps}" \
        "train.batch_size=${batch}" \
        "train.num_workers=${workers}" \
        "out_dir=${out_dir}" \
        "${extra[@]}"
}

pipeline_probe() {
    local model="$1"
    local max_samples="${2:-8192}"
    local ckpt="outputs/pusht_${model}_seed0/model.pt"

    pipeline_banner "Probe ${model}"
    python3 scripts/probe.py \
        "checkpoint=${ckpt}" \
        synthetic_fallback=false \
        "probe.max_samples=${max_samples}"
}

pipeline_plot() {
    pipeline_banner "Plot results"
    local probe_files=()
    local eval_files=()

    for f in outputs/probe/probe_pusht_factored_seed0.json \
             outputs/probe/probe_pusht_monolithic_seed0.json; do
        if [[ -f "$f" ]]; then
            probe_files+=("$f")
        fi
    done

    for f in outputs/eval/eval_pusht_factored_seed0_mppi.json \
             outputs/eval/eval_pusht_monolithic_seed0_mppi.json; do
        if [[ -f "$f" ]]; then
            eval_files+=("$f")
        fi
    done

    if [[ ${#probe_files[@]} -eq 0 && ${#eval_files[@]} -eq 0 ]]; then
        echo "[warn] No probe or eval JSON files found to plot."
        return 0
    fi

    local cmd=(python3 scripts/plot_results.py)
    if [[ ${#probe_files[@]} -gt 0 ]]; then
        cmd+=(--probe "${probe_files[@]}")
    fi
    if [[ ${#eval_files[@]} -gt 0 ]]; then
        cmd+=(--eval "${eval_files[@]}")
    fi
    "${cmd[@]}"
    echo "[done] Figures saved under outputs/figures/"
}

pipeline_summary() {
    pipeline_banner "Summary"
    echo "Probe (factored):  outputs/probe/probe_pusht_factored_seed0.json"
    echo "Probe (monolithic): outputs/probe/probe_pusht_monolithic_seed0.json"
    echo "Eval (factored):   outputs/eval/eval_pusht_factored_seed0_mppi.json"
    echo "Eval (monolithic): outputs/eval/eval_pusht_monolithic_seed0_mppi.json"
    echo "Figures:           outputs/figures/probe_r2.png"
}
