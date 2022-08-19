#!/usr/bin/env bash

# from readme:
# If your dataset is organized with one sample per line, you can
# use the `--line_by_line` flag (otherwise the script concatenates all
# texts and then splits them in blocks of the same length).

python3 run_mlm.py \
    --model_type bert \
    --config_name config.json \
    --tokenizer_name Tokenizer \
    --train_file notes.txt \
    --max_seq_length 512 \
    --per_device_train_batch_size 28 \
    --per_device_eval_batch_size 28 \
    --do_train \
    --do_eval \
    --output_dir Output \
    --overwrite_output_dir \
    --learning_rate 5e-05 \
    --num_train_epochs 100 \
    --max_steps -1 \
    --log_level passive \
    --logging_dir Logs \
    --save_steps 50000 \
    --disable_tqdm True \
    --logging_strategy epoch \
    --evaluation_strategy epoch
