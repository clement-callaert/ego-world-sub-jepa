#!/usr/bin/env bash
# Medium pipeline for a first meaningful run (~30 to 90 minutes on GPU).
#
# Usage:
#   bash scripts/pipeline_medium.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"
pipeline_root

EPISODES=500
COLLECT_PROCESSES=4
TRAIN_STEPS=5000
TRAIN_BATCH=128
TRAIN_WORKERS=4
EVAL_EPISODES=20
EVAL_MAX_STEPS=150
EVAL_SAMPLES=128
EVAL_ITERS=3

pipeline_banner "PushT pipeline (MEDIUM)"
echo "Collect: ${EPISODES} episodes"
echo "Train:   ${TRAIN_STEPS} steps x factored + monolithic"
echo "Eval:    ${EVAL_EPISODES} episodes, MPPI n_samples=${EVAL_SAMPLES} n_iters=${EVAL_ITERS}"

pipeline_banner "Collect data (${EPISODES} episodes)"
python3 scripts/collect_data.py \
    --episodes "${EPISODES}" \
    --out data/pusht.lance \
    --processes "${COLLECT_PROCESSES}" \
    --num-envs 2 \
    --overwrite

pipeline_train factored "${TRAIN_STEPS}" "${TRAIN_BATCH}" "${TRAIN_WORKERS}"
pipeline_train monolithic "${TRAIN_STEPS}" "${TRAIN_BATCH}" "${TRAIN_WORKERS}"

pipeline_probe factored
pipeline_probe monolithic

_eval_common=(
    "episodes=${EVAL_EPISODES}"
    "max_episode_steps=${EVAL_MAX_STEPS}"
    "planner.n_samples=${EVAL_SAMPLES}"
    "planner.n_iters=${EVAL_ITERS}"
)

pipeline_banner "Eval factored"
python3 scripts/evaluate.py \
    checkpoint=outputs/pusht_factored_seed0/model.pt \
    "${_eval_common[@]}"

pipeline_banner "Eval monolithic"
python3 scripts/evaluate.py \
    checkpoint=outputs/pusht_monolithic_seed0/model.pt \
    "${_eval_common[@]}"

pipeline_plot
pipeline_summary
