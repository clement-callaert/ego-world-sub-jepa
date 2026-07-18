#!/usr/bin/env bash
# Multi-seed screening grid under a global wall-clock budget.
#
# Priority queue (exact order):
#   P1: seed 1 on g1, g3, g7, g2
#   P2: seed 1 on g4, g5, g6, g8
#   P3: seed 2 on g1, g3, g7, g2
#   P4: seed 2 on g4, g5, g6, g8
#
# Per run: train -> probe (abs+disp) -> diagnose -> MPPI 50 ep (shared detector).
# Resumable: complete checkpoint skips train; existing JSON skips that step.
# Never overwrites committed seed-0 JSON under results/grid/gN_*.json.
#
# Usage:
#   bash scripts/run_seeds.sh
#   BUDGET_HOURS=6 SMOKE_ONLY=1 bash scripts/run_seeds.sh
#   SKIP_SMOKE=1 START_FROM=g3:1 bash scripts/run_seeds.sh

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"
pipeline_root

TRAIN_STEPS="${TRAIN_STEPS:-20000}"
TRAIN_BATCH_96="${TRAIN_BATCH_96:-256}"
TRAIN_WORKERS="${TRAIN_WORKERS:-16}"
TRAIN_COMPILE="${TRAIN_COMPILE:-false}"
EVAL_COMPILE="${EVAL_COMPILE:-true}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
PROTOCOL_SEED="${PROTOCOL_SEED:-0}"
BUDGET_HOURS="${BUDGET_HOURS:-6}"
RUN_TIMEOUT_SEC="${RUN_TIMEOUT_SEC:-21600}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
START_FROM="${START_FROM:-}"

DATA_PATH="data/pusht_96.lance"
DETECTOR_PT="outputs/shared_pusht96_seed0/detector.pt"
SESSION_LOG="results/grid/session_log.txt"
SESSION_START_FILE="results/grid/session_start_epoch.txt"

GRID_NAMES=(
    g1_factored_sgT_cov025_aux1
    g2_monolithic_sgT_cov025_aux1
    g3_monolithic_sgF_cov0_aux1
    g4_factored_sgF_cov0_aux1
    g5_factored_sgT_cov0_aux1
    g6_factored_sgF_cov025_aux1
    g7_factored_sgT_cov025_aux0
    g8_monolithic_sgT_cov025_aux0
)

QUEUE=(
    g1:1 g3:1 g7:1 g2:1
    g4:1 g5:1 g6:1 g8:1
    g1:2 g3:2 g7:2 g2:2
    g4:2 g5:2 g6:2 g8:2
)

name_for_gid() {
    local gid="$1" n
    for n in "${GRID_NAMES[@]}"; do
        if [[ "${n%%_*}" == "${gid}" ]]; then
            echo "${n}"
            return 0
        fi
    done
    return 1
}

log_session() {
    mkdir -p "$(dirname "${SESSION_LOG}")"
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $*" | tee -a "${SESSION_LOG}"
}

budget_remaining_s() {
    local start now
    start=$(cat "${SESSION_START_FILE}")
    now=$(date +%s)
    echo $(( start + BUDGET_HOURS * 3600 - now ))
}

ckpt_complete() {
    local ckpt="$1"
    [[ -f "${ckpt}" ]] || return 1
    python3 - "${ckpt}" "${TRAIN_STEPS}" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sys.exit(0 if int(ckpt.get("step", 0)) >= int(sys.argv[2]) else 1)
PY
}

hash_init_params() {
    local model_name="$1" seed="$2"
    python3 - "${model_name}" "${seed}" <<'PY'
import hashlib, sys
from omegaconf import OmegaConf
from ewjepa import EgoWorldConfig, EgoWorldJEPA
from ewjepa.utils import set_seed
name, seed = sys.argv[1], int(sys.argv[2])
set_seed(seed)
cfg = OmegaConf.load(f"configs/model/grid/{name}.yaml")
d = OmegaConf.to_container(cfg, resolve=True)
model = EgoWorldJEPA(EgoWorldConfig(**d))
h = hashlib.sha256()
for p in model.parameters():
    h.update(p.detach().cpu().numpy().tobytes())
    break
print(h.hexdigest()[:16])
PY
}

run_cmd() {
    # Run a command, optionally under timeout. Returns the exit code.
    if command -v timeout >/dev/null 2>&1; then
        timeout --kill-after=60 "${RUN_TIMEOUT_SEC}" "$@"
        return $?
    fi
    "$@"
    return $?
}

copy_json() {
    local src="$1" dst="$2"
    mkdir -p "$(dirname "${dst}")"
    if [[ -f "${dst}" ]]; then
        echo "[skip] ${dst} exists"
        return 0
    fi
    if [[ -f "${src}" ]]; then
        cp "${src}" "${dst}"
        echo "[copy] ${src} -> ${dst}"
        return 0
    fi
    echo "[error] expected ${src} but missing" >&2
    return 1
}

[[ -d "${DATA_PATH}" ]] || { echo "[error] ${DATA_PATH} missing" >&2; exit 1; }
[[ -f "${DETECTOR_PT}" ]] || { echo "[error] ${DETECTOR_PT} missing" >&2; exit 1; }

mkdir -p results/grid outputs/grid outputs/probe outputs/diagnostics outputs/eval
if [[ ! -f "${SESSION_START_FILE}" ]]; then
    date +%s > "${SESSION_START_FILE}"
    log_session "SESSION_START budget_hours=${BUDGET_HOURS} train_compile=${TRAIN_COMPILE} eval_compile=${EVAL_COMPILE} workers=${TRAIN_WORKERS}"
else
    log_session "SESSION_RESUME remaining_s=$(budget_remaining_s)"
fi

pipeline_banner "Multi-seed grid budget=${BUDGET_HOURS}h steps=${TRAIN_STEPS} episodes=${EVAL_EPISODES}"

run_one() {
    local gid="$1" seed="$2"
    local name out_dir ckpt tag results_dir diag_dir
    local probe_out diag_out eval_out
    local run_start t0 status=OK rc=0 rem init_hash

    name="$(name_for_gid "${gid}")" || {
        log_session "SKIP ${gid} seed${seed}: unknown gid"
        return 0
    }
    if [[ ! -f "configs/model/grid/${name}.yaml" ]]; then
        log_session "SKIP ${gid} seed${seed}: missing yaml"
        return 0
    fi

    out_dir="outputs/grid/${name}_seed${seed}"
    ckpt="${out_dir}/model.pt"
    tag="$(basename "${out_dir}")"
    results_dir="results/grid/seed${seed}"
    diag_dir="results/diagnostics/grid/seed${seed}"
    mkdir -p "${out_dir}" "${results_dir}" "${diag_dir}"

    probe_out="outputs/probe/probe_${tag}.json"
    diag_out="outputs/diagnostics/diagnostics_${tag}.json"
    eval_out="outputs/eval/eval_${tag}_mppi.json"

    rem=$(budget_remaining_s)
    if (( rem < 600 )); then
        log_session "SKIP ${gid} seed${seed}: budget exhausted remaining_s=${rem}"
        return 2
    fi

    # Already complete?
    if [[ -f "${results_dir}/${gid}_probe.json" && -f "${results_dir}/${gid}_diagnostics.json" && -f "${results_dir}/${gid}_mppi.json" ]] && ckpt_complete "${ckpt}"; then
        log_session "SKIP ${gid} seed${seed}: already complete"
        return 0
    fi

    pipeline_banner "${gid} seed${seed} -> ${out_dir}"
    run_start=$(date +%s)
    log_session "START config=${gid} seed=${seed} out_dir=${out_dir} budget_remaining_s=${rem}"

    if init_hash=$(hash_init_params "${name}" "${seed}" 2>/dev/null); then
        local hash0=""
        hash0=$(hash_init_params "${name}" 0 2>/dev/null || true)
        log_session "INIT_HASH config=${gid} seed=${seed} hash=${init_hash} seed0_hash=${hash0}"
        if [[ -n "${hash0}" && "${init_hash}" == "${hash0}" ]]; then
            log_session "FAILED config=${gid} seed=${seed} reason=init_hash_equals_seed0"
            return 1
        fi
    else
        log_session "WARN config=${gid} seed=${seed}: init hash failed (non-fatal)"
    fi

    # train (resume from partial checkpoint when present)
    if ckpt_complete "${ckpt}"; then
        echo "[skip] checkpoint ${ckpt} complete"
    else
        local resume_flag="false"
        if [[ -f "${ckpt}" ]]; then
            resume_flag="true"
            log_session "RESUME config=${gid} seed=${seed} from existing ${ckpt}"
        fi
        t0=$(date +%s)
        set +e
        run_cmd python3 scripts/train.py \
            "model=grid/${name}" \
            "data=pusht_96" \
            "seed=${seed}" \
            "train.steps=${TRAIN_STEPS}" \
            "train.batch_size=${TRAIN_BATCH_96}" \
            "train.num_workers=${TRAIN_WORKERS}" \
            "train.warmup_steps=1000" \
            "train.compile=${TRAIN_COMPILE}" \
            "train.resume=${resume_flag}" \
            "out_dir=${out_dir}"
        rc=$?
        set -e
        if (( rc == 124 )); then
            log_session "FAILED config=${gid} seed=${seed} reason=train_timeout"
            status="FAILED"
        elif (( rc != 0 )); then
            log_session "FAILED config=${gid} seed=${seed} reason=train_exit_${rc}"
            status="FAILED"
        else
            log_session "TIMING config=${gid} seed=${seed} train_s=$(( $(date +%s) - t0 ))"
        fi
    fi

    # Remaining wall for this run
    local left=$(( RUN_TIMEOUT_SEC - ( $(date +%s) - run_start ) ))
    if [[ "${status}" == "OK" ]] && (( left < 60 )); then
        log_session "FAILED config=${gid} seed=${seed} reason=run_timeout_before_probe"
        status="FAILED"
    fi

    if [[ "${status}" == "OK" ]]; then
        if [[ -f "${probe_out}" ]]; then
            echo "[skip] ${probe_out}"
        else
            t0=$(date +%s)
            set +e
            python3 scripts/probe.py "checkpoint=${ckpt}" data=pusht_96 \
                seed="${PROTOCOL_SEED}" synthetic_fallback=false probe.max_samples=8192
            rc=$?
            set -e
            if (( rc != 0 )); then
                log_session "FAILED config=${gid} seed=${seed} reason=probe_exit_${rc}"
                status="FAILED"
            else
                log_session "TIMING config=${gid} seed=${seed} probe_s=$(( $(date +%s) - t0 ))"
            fi
        fi
    fi

    if [[ "${status}" == "OK" ]]; then
        if [[ -f "${diag_out}" ]]; then
            echo "[skip] ${diag_out}"
        else
            t0=$(date +%s)
            set +e
            python3 scripts/diagnose.py "checkpoint=${ckpt}" data=pusht_96 seed="${PROTOCOL_SEED}"
            rc=$?
            set -e
            if (( rc != 0 )); then
                log_session "FAILED config=${gid} seed=${seed} reason=diagnose_exit_${rc}"
                status="FAILED"
            else
                log_session "TIMING config=${gid} seed=${seed} diagnose_s=$(( $(date +%s) - t0 ))"
            fi
        fi
    fi

    left=$(( RUN_TIMEOUT_SEC - ( $(date +%s) - run_start ) ))
    if [[ "${status}" == "OK" ]] && (( left < 60 )); then
        log_session "FAILED config=${gid} seed=${seed} reason=run_timeout_before_eval"
        status="FAILED"
    fi

    if [[ "${status}" == "OK" ]]; then
        if [[ -f "${eval_out}" ]]; then
            echo "[skip] ${eval_out}"
        else
            t0=$(date +%s)
            set +e
            # Cap eval by remaining per-run budget
            RUN_TIMEOUT_SEC=${left} run_cmd python3 scripts/evaluate.py \
                "checkpoint=${ckpt}" data=pusht_96 \
                "block_detector=${DETECTOR_PT}" "episodes=${EVAL_EPISODES}" \
                "seed=${PROTOCOL_SEED}" "compile=${EVAL_COMPILE}"
            rc=$?
            set -e
            if (( rc == 124 )); then
                log_session "FAILED config=${gid} seed=${seed} reason=eval_timeout"
                status="FAILED"
            elif (( rc != 0 )); then
                log_session "FAILED config=${gid} seed=${seed} reason=eval_exit_${rc}"
                status="FAILED"
            else
                log_session "TIMING config=${gid} seed=${seed} eval_s=$(( $(date +%s) - t0 ))"
            fi
        fi
    fi

    if [[ "${status}" == "OK" ]]; then
        set +e
        copy_json "${probe_out}" "${results_dir}/${gid}_probe.json" && \
        copy_json "${diag_out}"  "${results_dir}/${gid}_diagnostics.json" && \
        copy_json "${eval_out}"  "${results_dir}/${gid}_mppi.json" && \
        copy_json "${diag_out}"  "${diag_dir}/${gid}_diagnostics.json"
        rc=$?
        set -e
        if (( rc != 0 )); then
            status="FAILED"
            log_session "FAILED config=${gid} seed=${seed} reason=copy_json"
        fi
    fi

    local dur=$(( $(date +%s) - run_start ))
    if [[ "${status}" == "OK" ]] && [[ ! -f "${results_dir}/${gid}_mppi.json" ]]; then
        status="FAILED"
        log_session "FAILED config=${gid} seed=${seed} reason=missing_json_artifacts"
    fi
    log_session "END config=${gid} seed=${seed} status=${status} duration_s=${dur} probe=${results_dir}/${gid}_probe.json diag=${results_dir}/${gid}_diagnostics.json mppi=${results_dir}/${gid}_mppi.json"
    [[ "${status}" == "OK" ]]
}

# Build effective queue
EFFECTIVE=()
started=0
for item in "${QUEUE[@]}"; do
    if [[ -n "${START_FROM}" && "${started}" -eq 0 ]]; then
        if [[ "${item}" != "${START_FROM}" ]]; then
            continue
        fi
        started=1
    fi
    EFFECTIVE+=("${item}")
done
if [[ "${SMOKE_ONLY}" == "1" ]]; then
    EFFECTIVE=(g1:1)
fi

log_session "QUEUE ${EFFECTIVE[*]}"

smoke_failed=0
first_item="${EFFECTIVE[0]:-}"
for item in "${EFFECTIVE[@]}"; do
    gid="${item%%:*}"
    seed="${item##*:}"
    rem=$(budget_remaining_s)
    if (( rem < 600 )); then
        log_session "STOP budget exhausted before ${gid} seed${seed} remaining_s=${rem}"
        break
    fi

    run_t0=$(date +%s)
    set +e
    run_one "${gid}" "${seed}"
    rc=$?
    set -e

    if [[ "${item}" == "g1:1" && "${SKIP_SMOKE}" != "1" ]]; then
        if (( rc != 0 )); then
            log_session "SMOKE_FAILED stopping queue"
            smoke_failed=1
            break
        fi
        log_session "SMOKE_OK measured_duration_s=$(( $(date +%s) - run_t0 )) remaining_s=$(budget_remaining_s)"
        if [[ "${SMOKE_ONLY}" == "1" ]]; then
            break
        fi
    fi
    if (( rc == 2 )); then
        break
    fi
done

if (( smoke_failed )); then
    log_session "SESSION_ABORT smoke failed"
    exit 1
fi

log_session "SESSION_QUEUE_DONE remaining_s=$(budget_remaining_s)"
echo "Session log: ${SESSION_LOG}"
echo "Next: python3 scripts/probe_perfactor.py && python3 scripts/aggregate_grid.py --multiseed"
