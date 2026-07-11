#!/usr/bin/env bash
# Reproduce the documented factored-hires planning pipeline.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=.

EPISODES="${EPISODES:-2000}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
TRAIN_BATCH="${TRAIN_BATCH:-256}"
DETECTOR_STEPS="${DETECTOR_STEPS:-6000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"

python3 scripts/collect_data.py --out data/pusht_96.lance \
    --episodes="${EPISODES}" --processes=32 --image-shape 96 96 --overwrite
python3 scripts/train.py model=factored_hires data=pusht_96 \
    train.steps="${TRAIN_STEPS}" train.batch_size="${TRAIN_BATCH}" \
    train.warmup_steps=1000 out_dir=outputs/pusht_hires_seed0
python3 scripts/train_detector.py --dataset data/pusht_96.lance \
    --out outputs/pusht_hires_seed0/detector.pt --img-size 96 \
    --steps="${DETECTOR_STEPS}"
python3 scripts/evaluate.py checkpoint=outputs/pusht_hires_seed0/model.pt \
    data=pusht_96 block_detector=outputs/pusht_hires_seed0/detector.pt \
    episodes="${EVAL_EPISODES}"
python3 scripts/copy_results.py \
    --planning-checkpoint=outputs/pusht_hires_seed0/model.pt \
    --block-detector=outputs/pusht_hires_seed0/detector.pt
