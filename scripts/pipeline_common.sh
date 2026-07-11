#!/usr/bin/env bash
# Shared helpers for pipeline_short / pipeline_medium / pipeline_long.
#
# Common overrides (examples):
#   TRAIN_COMPILE=1 TRAIN_STEPS=50000 bash scripts/pipeline_long.sh
#   DETECTOR_STEPS=8000 EVAL_EPISODES=100 bash scripts/pipeline_long.sh

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

# Exit 0 when a Lance dataset exists and the current user can read it.
pipeline_lance_readable() {
    local path="$1"
    python3 - <<PY
import os
import sys
from pathlib import Path

path = Path("${path}")
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

# Collect 64x64 PushT data when missing or FORCE_COLLECT=1.
pipeline_ensure_pusht_data() {
    local episodes="$1"
    local processes="$2"
    pipeline_ensure_lance_data "data/pusht.lance" "${episodes}" "${processes}" 64 64
}

# Collect Lance data when missing, unreadable, or FORCE_COLLECT=1.
pipeline_ensure_lance_data() {
    local out_path="$1"
    local episodes="$2"
    local processes="$3"
    local img_h="${4:-64}"
    local img_w="${5:-64}"
    local need_collect=0

    if [[ ! -d "${out_path}" ]] || [[ "${FORCE_COLLECT:-0}" == "1" ]]; then
        need_collect=1
    elif ! pipeline_lance_readable "${out_path}"; then
        echo "[warn] ${out_path} exists but is not readable; recollecting"
        need_collect=1
    fi

    if [[ "${need_collect}" == "1" ]]; then
        pipeline_banner "Collect data (${episodes} ep -> ${out_path}, ${img_h}x${img_w})"
        python3 scripts/collect_data.py \
            --episodes "${episodes}" \
            --out "${out_path}" \
            --processes "${processes}" \
            --num-envs 2 \
            --image-shape "${img_h}" "${img_w}" \
            --overwrite
    else
        echo "[skip] ${out_path} exists and is readable (set FORCE_COLLECT=1 to recollect)"
    fi
}

# Default checkpoint dir for a Hydra model/data pair.
pipeline_default_out_dir() {
    local model="$1"
    case "${model}" in
        factored) echo "outputs/pusht_factored_seed0" ;;
        monolithic) echo "outputs/pusht_monolithic_seed0" ;;
        factored_hires) echo "outputs/pusht_hires_seed0" ;;
        *) echo "outputs/pusht_${model}_seed0" ;;
    esac
}

pipeline_train() {
    local model="$1"
    local steps="$2"
    local batch="$3"
    local workers="$4"
    local data="${5:-pusht}"
    local out_dir="${6:-$(pipeline_default_out_dir "${model}")}"
    local max_episodes="${7:-}"
    local compile="${8:-${TRAIN_COMPILE:-0}}"

    pipeline_banner "Train ${model} (${steps} steps) -> ${out_dir}"
    local extra=()
    if [[ -n "${max_episodes}" ]]; then
        extra+=("data.max_episodes=${max_episodes}")
    fi
    if [[ "${compile}" == "1" ]]; then
        extra+=("train.compile=true")
        echo "[info] torch.compile enabled (first steps are slower while compiling)"
    fi
    if [[ "${model}" == "factored_hires" ]]; then
        extra+=("train.warmup_steps=${TRAIN_WARMUP:-1000}")
    fi

    python3 scripts/train.py \
        "model=${model}" \
        "data=${data}" \
        "train.steps=${steps}" \
        "train.batch_size=${batch}" \
        "train.num_workers=${workers}" \
        "out_dir=${out_dir}" \
        "${extra[@]}"
}

pipeline_train_detector() {
    local dataset="${1:-data/pusht_96.lance}"
    local out_path="${2:-outputs/pusht_hires_seed0/detector.pt}"
    local img_size="${3:-96}"
    local steps="${4:-${DETECTOR_STEPS:-6000}}"

    pipeline_banner "Train block detector (${steps} steps) -> ${out_path}"
    python3 scripts/train_detector.py \
        --dataset "${dataset}" \
        --out "${out_path}" \
        --img-size "${img_size}" \
        --steps "${steps}"
}

pipeline_probe() {
    local ckpt="$1"
    local max_samples="${2:-8192}"
    local data="${3:-pusht}"

    pipeline_banner "Probe ${ckpt}"
    python3 scripts/probe.py \
        "checkpoint=${ckpt}" \
        "data=${data}" \
        synthetic_fallback=false \
        "probe.max_samples=${max_samples}"
}

# Full planning eval (optionally with block detector).
pipeline_eval() {
    local ckpt="$1"
    shift
    pipeline_banner "Eval ${ckpt}"
    python3 scripts/evaluate.py "checkpoint=${ckpt}" "$@"
}

pipeline_plot() {
    pipeline_banner "Plot results"
    local probe_files=()
    local eval_files=()

    while IFS= read -r -d '' f; do
        probe_files+=("$f")
    done < <(find outputs/probe -maxdepth 1 -name 'probe_*.json' -print0 2>/dev/null || true)

    while IFS= read -r -d '' f; do
        eval_files+=("$f")
    done < <(find outputs/eval -maxdepth 1 -name 'eval_*.json' -print0 2>/dev/null || true)

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
    echo "Best planning path (current):"
    echo "  World model:  outputs/pusht_hires_seed0/model.pt"
    echo "  Detector:     outputs/pusht_hires_seed0/detector.pt"
    echo "  Eval JSON:    outputs/eval/eval_pusht_hires_seed0_mppi.json"
    echo ""
    echo "64px baselines:"
    echo "  Factored:     outputs/pusht_factored_seed0/model.pt"
    echo "  Monolithic:   outputs/pusht_monolithic_seed0/model.pt"
    echo ""
    echo "Speed tip: TRAIN_COMPILE=1 on train.py (torch.compile, slower first steps)."
    echo "Long run:   TRAIN_STEPS=50000 DETECTOR_STEPS=8000 EVAL_EPISODES=100 bash scripts/pipeline_long.sh"
}
