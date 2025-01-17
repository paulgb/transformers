#!/bin/bash

/data/anaconda/envs/pyt/bin/python ./examples/run_squad.py \
    --model_type bert \
    --model_name_or_path /home/cs/models/new-attention-20200307/ \
    --do_eval \
    --do_lower_case \
    --predict_file /home/cs/cs224n-project/data/dev-v2.0.json \
    --max_seq_length 384 \
    --doc_stride 128 \
    --version_2_with_negative \
    --output_dir /home/cs/models/model/ \
    --per_gpu_eval_batch_size=1 > ~/attention-new.json
