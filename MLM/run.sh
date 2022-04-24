#!/usr/bin/env bash

python3 run_mlm.py \
    --model_type bert \
    --config_name bert-base-uncased \
    --tokenizer_name bert-base-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --output_dir Output \
    --overwrite_output_dir

