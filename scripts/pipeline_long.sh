#!/usr/bin/env bash
# Full pipeline for the best planning score (several hours on GPU).
#
# Trains factored_hires at 96px, trains the block detector, runs full MPPI eval.
# Also trains 64px factored + monolithic for probe comparison.
#
# Usage:
#   bash scripts/pipeline_long.sh
#
# Longer / faster training:
#   TRAIN_COMPILE=1 TRAIN_STEPS=50000 bash scripts/pipeline_long.sh
#   DETECTOR_STEPS=8000 EVAL_EPISODES=100 bash scripts/pipeline_long.sh
#
# At 96x96 use batch 256 (~19 GB). Batch 512 may OOM on 32 GB.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"
pipeline_root

EPISODES="${EPISODES:-2000}"
COLLECT_PROCESSES="${COLLECT_PROCESSES:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
TRAIN_BATCH_64="${TRAIN_BATCH_64:-128}"
TRAIN_BATCH_96="${TRAIN_BATCH_96:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
DETECTOR_STEPS="${DETECTOR_STEPS:-6000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
RUN_ROBUSTNESS="${RUN_ROBUSTNESS:-0}"
TRAIN_64="${TRAIN_64:-1}"

FACTORED_CKPT="outputs/pusht_factored_seed0/model.pt"
MONO_CKPT="outputs/pusht_monolithic_seed0/model.pt"
HIRES_CKPT="outputs/pusht_hires_seed0/model.pt"
DETECTOR_PT="outputs/pusht_hires_seed0/detector.pt"

pipeline_banner "PushT pipeline (LONG)"
echo "Collect:    ${EPISODES} episodes x 64px + 96px"
echo "Train 64px: ${TRAIN_64} (${TRAIN_STEPS} steps, batch ${TRAIN_BATCH_64})"
echo "Train 96px: factored_hires (${TRAIN_STEPS} steps, batch ${TRAIN_BATCH_96})"
echo "Detector:   ${DETECTOR_STEPS} steps"
echo "Eval:       ${EVAL_EPISODES} episodes, full MPPI + detector"
echo "Robustness: ${RUN_ROBUSTNESS}"
echo "Compile:    ${TRAIN_COMPILE:-0} (torch.compile via train.compile=true)"

pipeline_ensure_pusht_data "${EPISODES}" "${COLLECT_PROCESSES}"
pipeline_ensure_lance_data "data/pusht_96.lance" "${EPISODES}" "${COLLECT_PROCESSES}" 96 96

if [[ "${TRAIN_64}" == "1" ]]; then
    pipeline_train factored "${TRAIN_STEPS}" "${TRAIN_BATCH_64}" "${TRAIN_WORKERS}"
    pipeline_train monolithic "${TRAIN_STEPS}" "${TRAIN_BATCH_64}" "${TRAIN_WORKERS}"
    pipeline_probe "${FACTORED_CKPT}"
    pipeline_probe "${MONO_CKPT}"
fi

pipeline_train factored_hires "${TRAIN_STEPS}" "${TRAIN_BATCH_96}" "${TRAIN_WORKERS}" pusht_96
pipeline_train_detector "data/pusht_96.lance" "${DETECTOR_PT}" 96 "${DETECTOR_STEPS}"

_eval_args=(
    "data=pusht_96"
    "block_detector=${DETECTOR_PT}"
    "episodes=${EVAL_EPISODES}"
)

if [[ "${RUN_ROBUSTNESS}" == "1" ]]; then
    _eval_args+=("robustness.enabled=true")
fi

pipeline_eval "${HIRES_CKPT}" "${_eval_args[@]}"

python3 scripts/copy_results.py \
    --planning-checkpoint="${HIRES_CKPT}" \
    --block-detector="${DETECTOR_PT}"

# Optional monolithic planning without a detector.
if [[ "${EVAL_MONOLITHIC:-0}" == "1" ]]; then
    pipeline_eval "${MONO_CKPT}" "episodes=${EVAL_EPISODES}"
fi

pipeline_plot
pipeline_summary
