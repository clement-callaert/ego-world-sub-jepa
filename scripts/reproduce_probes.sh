#!/usr/bin/env bash
# Reproduce the archived Jul-6 probe configurations.
#
# This writes new outputs. Matching the committed metrics also requires the
# original dataset, software environment, and random seed.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=.

TRAIN_STEPS="${TRAIN_STEPS:-20000}"
TRAIN_BATCH="${TRAIN_BATCH:-128}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"

python3 scripts/train.py model=factored_cov data=pusht \
    train.steps="${TRAIN_STEPS}" train.batch_size="${TRAIN_BATCH}" \
    train.num_workers="${TRAIN_WORKERS}" \
    out_dir=outputs/pusht_factored_cov_seed0
python3 scripts/probe.py checkpoint=outputs/pusht_factored_cov_seed0/model.pt \
    synthetic_fallback=false probe.num_steps=9 probe.max_samples=8192

python3 scripts/train.py model=monolithic_cov data=pusht \
    train.steps="${TRAIN_STEPS}" train.batch_size="${TRAIN_BATCH}" \
    train.num_workers="${TRAIN_WORKERS}" \
    out_dir=outputs/pusht_monolithic_seed0
python3 scripts/probe.py checkpoint=outputs/pusht_monolithic_seed0/model.pt \
    synthetic_fallback=false probe.num_steps=9 probe.max_samples=8192

python3 scripts/copy_results.py \
    --factored-checkpoint=outputs/pusht_factored_cov_seed0/model.pt \
    --monolithic-checkpoint=outputs/pusht_monolithic_seed0/model.pt
