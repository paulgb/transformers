
/data/anaconda/envs/squad/bin/python ./examples/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --do_eval \
    --do_lower_case \
    --predict_file /home/cs/cs224n-project/data/dev-v2.0.json \
    --max_seq_length 384 \
    --doc_stride 128 \
    --version_2_with_negative \
    --output_dir /home/cs/models/model \
    --eval_all_checkpoints \
    --per_gpu_eval_batch_size=1
