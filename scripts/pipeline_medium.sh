#!/usr/bin/env bash
# Medium pipeline (~45 to 120 minutes on GPU).
# Trains 64px baselines and runs a light planning eval (no detector).
#
# Usage:
#   bash scripts/pipeline_medium.sh
#   TRAIN_COMPILE=1 bash scripts/pipeline_medium.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"
pipeline_root

EPISODES=500
COLLECT_PROCESSES=4
TRAIN_STEPS=5000
TRAIN_BATCH=128
TRAIN_WORKERS=4
EVAL_EPISODES=20
EVAL_MAX_STEPS=300
EVAL_SAMPLES=256
EVAL_ITERS=4

FACTORED_CKPT="outputs/pusht_factored_seed0/model.pt"
MONO_CKPT="outputs/pusht_monolithic_seed0/model.pt"

pipeline_banner "PushT pipeline (MEDIUM)"
echo "Collect: ${EPISODES} episodes (64px)"
echo "Train:   ${TRAIN_STEPS} steps x factored + monolithic"
echo "Eval:    ${EVAL_EPISODES} ep, MPPI n_samples=${EVAL_SAMPLES} n_iters=${EVAL_ITERS} (no detector)"
echo "Compile: ${TRAIN_COMPILE:-0}"

pipeline_ensure_pusht_data "${EPISODES}" "${COLLECT_PROCESSES}"

pipeline_train factored "${TRAIN_STEPS}" "${TRAIN_BATCH}" "${TRAIN_WORKERS}"
pipeline_train monolithic "${TRAIN_STEPS}" "${TRAIN_BATCH}" "${TRAIN_WORKERS}"

pipeline_probe "${FACTORED_CKPT}"
pipeline_probe "${MONO_CKPT}"

_eval_common=(
    "episodes=${EVAL_EPISODES}"
    "max_episode_steps=${EVAL_MAX_STEPS}"
    "planner.n_samples=${EVAL_SAMPLES}"
    "planner.n_iters=${EVAL_ITERS}"
)

pipeline_eval "${FACTORED_CKPT}" "${_eval_common[@]}"
pipeline_eval "${MONO_CKPT}" "${_eval_common[@]}"

pipeline_plot
pipeline_summary
