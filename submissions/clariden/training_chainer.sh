#!/bin/bash
MAX_RUNS=3
SCRIPT=training.sh

START_RUN=${1:-1}

# First job
jid=$(sbatch --parsable --export=ALL,RUN_NUM=$START_RUN $SCRIPT)
echo "Submitting run $START_RUN with job $jid"

# Chain the rest
if [ $((START_RUN + 1)) -le $MAX_RUNS ]; then
    for run in $(seq $((START_RUN + 1)) $MAX_RUNS); do
        echo "Submitting run $run after job $jid"
        jid=$(sbatch --parsable --dependency=afterok:$jid --export=ALL,RUN_NUM=$run $SCRIPT)
    done
fi
