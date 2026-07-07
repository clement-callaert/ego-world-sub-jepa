#!/usr/bin/env bash
# Reproduce canonical results: train (cov on), probe, eval, copy to results/.
#
# Usage:
#   bash scripts/reproduce.sh              # full 20k pipeline
#   TRAIN_STEPS=5000 bash scripts/reproduce.sh   # faster smoke reproduction
#
# Requires data/pusht.lance (see scripts/collect_data.py).

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"
pipeline_root

TRAIN_STEPS="${TRAIN_STEPS:-20000}"
TRAIN_BATCH="${TRAIN_BATCH:-128}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
PROBE_MAX_SAMPLES="${PROBE_MAX_SAMPLES:-50000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
# The default factored config now trains with state supervision (state_aux_weight).
FACTORED_OUT="outputs/pusht_factored_stateaux_seed0"
MONO_OUT="outputs/pusht_monolithic_seed0"

pipeline_banner "Reproduce: factored (state supervision) ${TRAIN_STEPS} steps"
pipeline_ensure_pusht_data "${EPISODES:-2000}" "${COLLECT_PROCESSES:-8}"

python3 scripts/train.py \
    model=factored \
    data=pusht \
    "train.steps=${TRAIN_STEPS}" \
    "train.batch_size=${TRAIN_BATCH}" \
    "train.num_workers=${TRAIN_WORKERS}" \
    "out_dir=${FACTORED_OUT}"

pipeline_banner "Probe factored + monolithic (max_samples=${PROBE_MAX_SAMPLES})"
python3 scripts/probe.py \
    "checkpoint=${FACTORED_OUT}/model.pt" \
    synthetic_fallback=false \
    "probe.max_samples=${PROBE_MAX_SAMPLES}"

if [[ -f "${MONO_OUT}/model.pt" ]]; then
    python3 scripts/probe.py \
        "checkpoint=${MONO_OUT}/model.pt" \
        synthetic_fallback=false \
        "probe.max_samples=${PROBE_MAX_SAMPLES}"
else
    pipeline_banner "Train monolithic baseline (${TRAIN_STEPS} steps)"
    pipeline_train monolithic "${TRAIN_STEPS}" "${TRAIN_BATCH}" "${TRAIN_WORKERS}"
    python3 scripts/probe.py \
        "checkpoint=${MONO_OUT}/model.pt" \
        synthetic_fallback=false \
        "probe.max_samples=${PROBE_MAX_SAMPLES}"
fi

pipeline_banner "Eval factored (${EVAL_EPISODES} episodes)"
python3 scripts/evaluate.py \
    "checkpoint=${FACTORED_OUT}/model.pt" \
    "episodes=${EVAL_EPISODES}"

pipeline_banner "Record rollout video (3 episodes)"
python3 scripts/record_video.py \
    "checkpoint=${FACTORED_OUT}/model.pt" \
    episodes=3 \
    "out_dir=outputs/videos/pusht_factored_stateaux_seed0"

pipeline_plot

pipeline_banner "Copy artifacts to results/"
python3 scripts/copy_results.py \
    --factored-checkpoint "${FACTORED_OUT}/model.pt" \
    --monolithic-checkpoint "${MONO_OUT}/model.pt"

pipeline_summary
echo ""
echo "Committed-ready artifacts under results/ (see results/manifest.json)"
