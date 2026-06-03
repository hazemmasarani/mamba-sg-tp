#!/bin/bash

NUM_ITER=200

run_case () {
    local BATCH=$1
    local SEQ=$2

    while true
    do
        echo "Running batch_size=$BATCH seq_len=$SEQ num_iter=$NUM_ITER"

        python run.py \
            -batch_size $BATCH \
            -seq_len $SEQ \
            -num_iter $NUM_ITER

        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "Case succeeded"
            break
        fi

        echo "run.py crashed with exit code $EXIT_CODE"
        echo "Killing GPU processes..."

        nvidia-smi --query-compute-apps=pid \
                   --format=csv,noheader \
        | xargs -r kill -9

        echo "Waiting 60 seconds before retry..."
        sleep 60
    done
}

run_experiments () {
    local BATCH=$1
    local MAX_SEQ=$2

    for ((SEQ=4; SEQ<=MAX_SEQ; SEQ*=2))
    do
        run_case $BATCH $SEQ
    done
}

run_experiments 1 1024
run_experiments 2 512
run_experiments 4 256
run_experiments 8 128
run_experiments 16 64
run_experiments 32 32
run_experiments 64 16