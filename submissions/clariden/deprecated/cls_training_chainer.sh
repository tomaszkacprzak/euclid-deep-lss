#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBES=("lensing" "clustering" "cross" "2x2pt" "combined")

for PROBE in "${PROBES[@]}"; do
    sbatch --export=ALL,PROBE="$PROBE" --job-name="cls_${PROBE}" "$SCRIPT_DIR/cls_training.sh"
    echo "Submitted cls_training for probe=$PROBE"
done
