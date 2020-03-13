#!/bin/bash

# --model_name_or_path bert-large-uncased-whole-word-masking \
#    --do_eval \
#    --evaluate_during_training

# /data/anaconda/envs/pyt/bin/python ./examples/run_squad.py \
# /data/anaconda/envs/pyt/bin/python -m examples.run_squad \
/data/anaconda/envs/pyt/bin/python -m torch.distributed.launch --nproc_per_node=2 ./examples/run_squad.py \
    --model_type bert2 \
    --model_name_or_path /mnt/models/new-attention-20200306-more-cycles/ \
    --do_train \
    --do_lower_case \
    --train_file /home/cs/cs224n-project/data/train-v2.0.json \
    --predict_file /home/cs/cs224n-project/data/dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 10 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --version_2_with_negative \
    --output_dir /mnt/models/new-attention-20200307/ \
    --per_gpu_eval_batch_size=8 \
    --per_gpu_train_batch_size=8 \
    --fp16 \
    --save_steps 2000

