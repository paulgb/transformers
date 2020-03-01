
/data/anaconda/envs/squad/bin/python ./examples/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file /home/cs/cs224n-project/data/train-v2.0.json \
    --predict_file /home/cs/cs224n-project/data/dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --version_2_with_negative \
    --output_dir /mnt/models/wwm_uncased_finetuned_squad/ \
    --per_gpu_eval_batch_size=1 \
    --per_gpu_train_batch_size=1 \
    --save_steps 20000 \
    --evaluate_during_training
