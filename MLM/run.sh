#!/usr/bin/env bash

python3 run_mlm.py \
    --model_type bert \
    --config_name bert-base-uncased \
    --tokenizer_name bert-base-uncased \
    --dataset_name bookcorpus \
    --dataset_config_name plain_text \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --output_dir Outq \
    --overwrite_output_dir
