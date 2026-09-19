#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "Starting Ablation Phase N..."
export MPLCONFIGDIR=/tmp/matplotlib_cache
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -u scripts/tuning/run_ablation.py

echo "Starting Auto-Selection and Hyperparameter Sweep Phase O..."
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -u scripts/tuning/auto_sweep.py

echo "ALL TUNING FINISHED"
