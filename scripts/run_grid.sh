#!/usr/bin/env bash
# Screening grid (Step 2): 8 configs under configs/model/grid/, seed 0.
#
# For each config: train (20k steps, batch 256, same hyperparameters as the
# Tier A runs), then probe, rollout/sensitivity diagnostics, MPPI eval with
# the existing SHARED detector, and copy the JSON artifacts to results/grid/.
#
# Resumable: a step is skipped when its output already exists. g1 and g3 are
# byte-identical (up to comments) to factored_hires and monolithic_hires, so
# they reuse the existing Tier A checkpoints and cost no training time.
#
# Budget: measured on the RTX 5090, one run is about 63 min train + 22 to 42
# min MPPI eval + 4 min diagnostics, well under the 6 h per-run limit.
# Do not launch while another training occupies the GPU (batch 256 at 96px
# needs about 19 GB).
#
# Usage:
#   bash scripts/run_grid.sh
#   TRAIN_STEPS=20000 EVAL_EPISODES=50 bash scripts/run_grid.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"
pipeline_root

TRAIN_STEPS="${TRAIN_STEPS:-20000}"
TRAIN_BATCH_96="${TRAIN_BATCH_96:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"

DATA_PATH="data/pusht_96.lance"
DETECTOR_PT="outputs/shared_pusht96_seed0/detector.pt"
RESULTS_DIR="results/grid"
TIMING_LOG="outputs/grid/timing.log"

[[ -d "${DATA_PATH}" ]] || { echo "[error] ${DATA_PATH} missing" >&2; exit 1; }
[[ -f "${DETECTOR_PT}" ]] || { echo "[error] ${DETECTOR_PT} missing (shared detector)" >&2; exit 1; }
mkdir -p "${RESULTS_DIR}" outputs/grid

GRID=(
    g1_factored_sgT_cov025_aux1
    g2_monolithic_sgT_cov025_aux1
    g3_monolithic_sgF_cov0_aux1
    g4_factored_sgF_cov0_aux1
    g5_factored_sgT_cov0_aux1
    g6_factored_sgF_cov025_aux1
    g7_factored_sgT_cov025_aux0
    g8_monolithic_sgT_cov025_aux0
)

# g1 and g3 reuse the Tier A checkpoints (identical configs).
out_dir_for() {
    case "$1" in
        g1_*) echo "outputs/pusht_hires_seed0" ;;
        g3_*) echo "outputs/pusht_monolithic_hires_seed0" ;;
        *)    echo "outputs/grid/$1_seed0" ;;
    esac
}

log_time() {
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $*" | tee -a "${TIMING_LOG}"
}

copy_if_missing() {
    local src="$1" dst="$2"
    if [[ -f "${dst}" ]]; then
        echo "[skip] ${dst} exists"
    elif [[ -f "${src}" ]]; then
        cp "${src}" "${dst}"
        echo "[copy] ${src} -> ${dst}"
    else
        echo "[error] expected ${src} but it is missing" >&2
        exit 1
    fi
}

pipeline_banner "Screening grid: ${#GRID[@]} configs, seed 0, ${TRAIN_STEPS} steps, ${EVAL_EPISODES} eval episodes"

for name in "${GRID[@]}"; do
    gid="${name%%_*}"
    out_dir="$(out_dir_for "${name}")"
    ckpt="${out_dir}/model.pt"
    tag="$(basename "${out_dir}")"

    pipeline_banner "${name} -> ${out_dir}"
    run_start=$(date +%s)

    # train (skipped when a COMPLETE checkpoint exists: resumable, and reuses
    # the Tier A checkpoints for g1/g3). train.py saves partial checkpoints
    # every 500 steps, so existence alone is not enough: check the step count.
    ckpt_complete() {
        [[ -f "$1" ]] || return 1
        python3 - "$1" "${TRAIN_STEPS}" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sys.exit(0 if int(ckpt.get("step", 0)) >= int(sys.argv[2]) else 1)
PY
    }
    if ckpt_complete "${ckpt}"; then
        echo "[skip] checkpoint ${ckpt} is complete, not retraining"
    else
        t0=$(date +%s)
        python3 scripts/train.py \
            "model=grid/${name}" \
            "data=pusht_96" \
            "train.steps=${TRAIN_STEPS}" \
            "train.batch_size=${TRAIN_BATCH_96}" \
            "train.num_workers=${TRAIN_WORKERS}" \
            "train.warmup_steps=1000" \
            "out_dir=${out_dir}"
        log_time "${gid} train_s=$(( $(date +%s) - t0 ))"
    fi

    # probe
    probe_out="outputs/probe/probe_${tag}.json"
    if [[ -f "${probe_out}" ]]; then
        echo "[skip] ${probe_out} exists"
    else
        t0=$(date +%s)
        python3 scripts/probe.py "checkpoint=${ckpt}" data=pusht_96 \
            synthetic_fallback=false probe.max_samples=8192
        log_time "${gid} probe_s=$(( $(date +%s) - t0 ))"
    fi

    # rollout error + action sensitivity
    diag_out="outputs/diagnostics/diagnostics_${tag}.json"
    if [[ -f "${diag_out}" ]]; then
        echo "[skip] ${diag_out} exists"
    else
        t0=$(date +%s)
        python3 scripts/diagnose.py "checkpoint=${ckpt}" data=pusht_96
        log_time "${gid} diagnose_s=$(( $(date +%s) - t0 ))"
    fi

    # MPPI eval with the shared detector
    eval_out="outputs/eval/eval_${tag}_mppi.json"
    if [[ -f "${eval_out}" ]]; then
        echo "[skip] ${eval_out} exists"
    else
        t0=$(date +%s)
        python3 scripts/evaluate.py "checkpoint=${ckpt}" data=pusht_96 \
            "block_detector=${DETECTOR_PT}" "episodes=${EVAL_EPISODES}"
        log_time "${gid} eval_s=$(( $(date +%s) - t0 ))"
    fi

    copy_if_missing "${probe_out}" "${RESULTS_DIR}/${gid}_probe.json"
    copy_if_missing "${diag_out}"  "${RESULTS_DIR}/${gid}_diagnostics.json"
    copy_if_missing "${eval_out}"  "${RESULTS_DIR}/${gid}_mppi.json"

    log_time "${gid} run_total_s=$(( $(date +%s) - run_start ))"
done

pipeline_banner "Grid complete"
echo "Artifacts in ${RESULTS_DIR}/ (gN_probe.json, gN_diagnostics.json, gN_mppi.json)"
echo "Timing log: ${TIMING_LOG}"
echo "Next: python3 scripts/aggregate_grid.py"
