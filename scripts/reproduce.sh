#!/usr/bin/env bash
# Backward-compatible entry point for the archived probe reproduction.
#
# Usage:
#   bash scripts/reproduce.sh
#   TRAIN_STEPS=5000 bash scripts/reproduce.sh
#
# Requires data/pusht.lance (see scripts/collect_data.py).

exec bash "$(dirname "${BASH_SOURCE[0]}")/reproduce_probes.sh"
