#!/usr/bin/env bash
# Controlled 96px comparison: factored_hires vs monolithic_hires on the same data.
#
# Runs the full pipeline (collect, train x2, shared detector, probes, MPPI eval x2)
# and archives JSON artifacts under results/ for reproducible reporting.
#
# Prerequisites:
#   pip install -e ".[dev,experiments]"
#   export PYTHONPATH=.
#   CUDA GPU recommended (several hours on RTX-class hardware).
#   Pin versions with requirements-results.txt after installing a CUDA PyTorch wheel.
#
# Usage:
#   bash scripts/reproduce_full_comparison.sh
#   FORCE_COLLECT=1 bash scripts/reproduce_full_comparison.sh
#   TRAIN_STEPS=50000 EVAL_EPISODES=100 bash scripts/reproduce_full_comparison.sh
#
# At 96x96 use batch 256 (~19 GB). Batch 512 may OOM on 32 GB.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"
pipeline_root

EPISODES="${EPISODES:-2000}"
COLLECT_PROCESSES="${COLLECT_PROCESSES:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
TRAIN_BATCH_96="${TRAIN_BATCH_96:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
DETECTOR_STEPS="${DETECTOR_STEPS:-6000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
RUN_ROBUSTNESS="${RUN_ROBUSTNESS:-0}"

DATA_PATH="data/pusht_96.lance"
FACTORED_CKPT="outputs/pusht_hires_seed0/model.pt"
MONO_CKPT="outputs/pusht_monolithic_hires_seed0/model.pt"
DETECTOR_DIR="outputs/shared_pusht96_seed0"
DETECTOR_PT="${DETECTOR_DIR}/detector.pt"
DETECTOR_METRICS="${DETECTOR_DIR}/detector_metrics.json"

pipeline_banner "Controlled comparison (96px): factored_hires vs monolithic_hires"
echo "Data:       ${DATA_PATH} (${EPISODES} episodes, FORCE_COLLECT=${FORCE_COLLECT:-0})"
echo "Train:      ${TRAIN_STEPS} steps, batch ${TRAIN_BATCH_96}"
echo "Detector:   ${DETECTOR_STEPS} steps -> ${DETECTOR_PT}"
echo "Eval:       ${EVAL_EPISODES} episodes, full MPPI + shared detector"
echo "Robustness: ${RUN_ROBUSTNESS}"
echo "Compile:    ${TRAIN_COMPILE:-0}"

pipeline_ensure_lance_data "${DATA_PATH}" "${EPISODES}" "${COLLECT_PROCESSES}" 96 96

pipeline_train factored_hires "${TRAIN_STEPS}" "${TRAIN_BATCH_96}" "${TRAIN_WORKERS}" pusht_96
pipeline_train monolithic_hires "${TRAIN_STEPS}" "${TRAIN_BATCH_96}" "${TRAIN_WORKERS}" pusht_96

mkdir -p "${DETECTOR_DIR}"
pipeline_train_detector "${DATA_PATH}" "${DETECTOR_PT}" 96 "${DETECTOR_STEPS}" "${DETECTOR_METRICS}"

pipeline_probe "${FACTORED_CKPT}" 8192 pusht_96
pipeline_probe "${MONO_CKPT}" 8192 pusht_96

_eval_args=(
    "data=pusht_96"
    "block_detector=${DETECTOR_PT}"
    "episodes=${EVAL_EPISODES}"
)

if [[ "${RUN_ROBUSTNESS}" == "1" ]]; then
    _eval_args+=("robustness.enabled=true")
fi

pipeline_eval "${FACTORED_CKPT}" "${_eval_args[@]}"
pipeline_eval "${MONO_CKPT}" "${_eval_args[@]}"

python3 scripts/copy_results.py \
    --factored-checkpoint="${FACTORED_CKPT}" \
    --monolithic-checkpoint="${MONO_CKPT}" \
    --block-detector="${DETECTOR_PT}" \
    --detector-metrics="${DETECTOR_METRICS}"

pipeline_plot

pipeline_banner "Controlled comparison summary"
echo "Shared data:     ${DATA_PATH}"
echo "Shared detector: ${DETECTOR_PT}"
echo "Factored hires:  ${FACTORED_CKPT}"
echo "  probe:         outputs/probe/probe_pusht_hires_seed0.json"
echo "  eval:          outputs/eval/eval_pusht_hires_seed0_mppi.json"
echo "Monolithic hires:${MONO_CKPT}"
echo "  probe:         outputs/probe/probe_pusht_monolithic_hires_seed0.json"
echo "  eval:          outputs/eval/eval_pusht_monolithic_hires_seed0_mppi.json"
echo ""
echo "Archived under results/ via copy_results.py (commit JSON only, not checkpoints)."
echo "Re-run: FORCE_COLLECT=1 bash scripts/reproduce_full_comparison.sh"
