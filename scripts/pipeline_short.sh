#!/usr/bin/env bash
# Quick sanity check (~5 to 15 minutes on GPU).
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

FACTORED_CKPT="outputs/pusht_factored_seed0/model.pt"
MONO_CKPT="outputs/pusht_monolithic_seed0/model.pt"

pipeline_banner "PushT pipeline (SHORT)"
echo "Collect: ${EPISODES} episodes (or skip if data exists)"
echo "Train:   ${TRAIN_STEPS} steps x factored + monolithic (64px)"
echo "Eval:    smoke (fast MPPI, no detector)"
echo "Compile: ${TRAIN_COMPILE:-0} (set TRAIN_COMPILE=1 to try torch.compile)"

pipeline_ensure_pusht_data "${EPISODES}" "${COLLECT_PROCESSES}"

pipeline_banner "Tests"
python3 -m pytest tests/ -q --ignore=tests/test_train_speed.py

pipeline_train factored "${TRAIN_STEPS}" "${TRAIN_BATCH}" "${TRAIN_WORKERS}" pusht "" "${EPISODES}"
pipeline_train monolithic "${TRAIN_STEPS}" "${TRAIN_BATCH}" "${TRAIN_WORKERS}" pusht "" "${EPISODES}"

pipeline_probe "${FACTORED_CKPT}" 8192
pipeline_probe "${MONO_CKPT}" 8192

pipeline_eval "${FACTORED_CKPT}" data.max_episodes="${EPISODES}" fast=true
pipeline_eval "${MONO_CKPT}" data.max_episodes="${EPISODES}" fast=true

pipeline_plot
pipeline_summary
