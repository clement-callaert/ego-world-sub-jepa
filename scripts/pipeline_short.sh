#!/usr/bin/env bash
# Quick end-to-end sanity check (~3 to 8 minutes on GPU).
#
# Usage:
#   bash scripts/pipeline_short.sh
#
# Skips data collection if data/pusht.lance already exists.
# Set FORCE_COLLECT=1 to recollect 50 episodes anyway.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"
pipeline_root

EPISODES=50
COLLECT_PROCESSES=2
TRAIN_STEPS=5000
TRAIN_BATCH=128
TRAIN_WORKERS=4

pipeline_banner "PushT pipeline (SHORT)"
echo "Collect: ${EPISODES} episodes (or skip if data exists)"
echo "Train:   ${TRAIN_STEPS} steps x factored + monolithic"
echo "Eval:    smoke (5 episodes, MPPI n_samples=128 iters=3, 300 steps)"

pipeline_ensure_pusht_data "${EPISODES}" "${COLLECT_PROCESSES}"

pipeline_banner "Tests"
python3 -m pytest tests/ -q --ignore=tests/test_train_speed.py

pipeline_train factored "${TRAIN_STEPS}" "${TRAIN_BATCH}" "${TRAIN_WORKERS}" "${EPISODES}"
pipeline_train monolithic "${TRAIN_STEPS}" "${TRAIN_BATCH}" "${TRAIN_WORKERS}" "${EPISODES}"

pipeline_probe factored 8192
pipeline_probe monolithic 8192

pipeline_banner "Smoke eval (factored)"
python3 scripts/evaluate.py \
    checkpoint=outputs/pusht_factored_seed0/model.pt \
    data.max_episodes="${EPISODES}" \
    fast=true

pipeline_banner "Smoke eval (monolithic)"
python3 scripts/evaluate.py \
    checkpoint=outputs/pusht_monolithic_seed0/model.pt \
    data.max_episodes="${EPISODES}" \
    fast=true

pipeline_plot
pipeline_summary
