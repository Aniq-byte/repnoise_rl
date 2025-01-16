#!/bin/sh

export TASKS="power"
echo "TASKS=$TASKS"

python3 -u train.py -v \
    --model gpt2 \
    --learning_rate 5e-6 \
    --dropout 0 \
    --ngpus 1 \
    --nepochs 2 \
    --max_length 256 \
    --batch_size 32 \
    --tasks $TASKS \
    --save