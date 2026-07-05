#!/usr/bin/env bash
# Full pipeline for converged planning experiments (several hours on GPU).
#
# Usage:
#   bash scripts/pipeline_long.sh
#
# For even longer runs, override train steps:
#   TRAIN_STEPS=50000 bash scripts/pipeline_long.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"
pipeline_root

EPISODES="${EPISODES:-2000}"
COLLECT_PROCESSES="${COLLECT_PROCESSES:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
TRAIN_BATCH="${TRAIN_BATCH:-128}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
EVAL_EPISODES=50
RUN_ROBUSTNESS="${RUN_ROBUSTNESS:-1}"

pipeline_banner "PushT pipeline (LONG)"
echo "Collect:    ${EPISODES} episodes (${COLLECT_PROCESSES} processes)"
echo "Train:      ${TRAIN_STEPS} steps x factored + monolithic"
echo "Eval:       ${EVAL_EPISODES} episodes, default MPPI budget"
echo "Robustness: ${RUN_ROBUSTNESS} (factored only)"

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

pipeline_banner "Eval factored (full MPPI)"
if [[ "${RUN_ROBUSTNESS}" == "1" ]]; then
    python3 scripts/evaluate.py \
        checkpoint=outputs/pusht_factored_seed0/model.pt \
        "episodes=${EVAL_EPISODES}" \
        robustness.enabled=true
else
    python3 scripts/evaluate.py \
        checkpoint=outputs/pusht_factored_seed0/model.pt \
        "episodes=${EVAL_EPISODES}"
fi

pipeline_banner "Eval monolithic (full MPPI)"
python3 scripts/evaluate.py \
    checkpoint=outputs/pusht_monolithic_seed0/model.pt \
    "episodes=${EVAL_EPISODES}"

pipeline_plot
pipeline_summary
