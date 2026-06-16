#!/bin/bash

set -e  # stop on first error

declare -A seq_lens

seq_lens[1]="4 8 16 32 64 128 256"
seq_lens[2]="4 8 16 32 64 128"
seq_lens[4]="4 8 16 32 64"
seq_lens[8]="4 8 16 32"
seq_lens[16]="4 8 16"

for batch_size in 1 2 4 8 16; do
    for seq_len in ${seq_lens[$batch_size]}; do
        echo "=================================================="
        echo "Running batch_size=${batch_size}, seq_len=${seq_len}"
        echo "=================================================="

        python run_1.py \
            -batch_size "$batch_size" \
            -seq_len "$seq_len" \
            -num_iter 1

        python comp_acc.py

        echo
    done
done

echo "All experiments completed."