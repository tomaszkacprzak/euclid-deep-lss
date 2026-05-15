#!/bin/bash
MAX_RUNS=${1:-3}
SCRIPT=resume_training.sh

# First job
jid=$(sbatch --parsable $SCRIPT)
echo "Submitting run 1 with job $jid"

# Chain the rest
for run in $(seq 2 $MAX_RUNS); do
    echo "Submitting run $run after job $jid"
    jid=$(sbatch --parsable --dependency=afterok:$jid $SCRIPT)
done
