#!/usr/bin/env bash

# comment param below to train from scratch \
# --model_name_or_path roberta-base \

python3 run_mlm.py \
    --model_type roberta \
    --config_name roberta-base \
    --tokenizer_name roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --do_train \
    --do_eval \
    --output_dir Output \
    --overwrite_output_dir
