#!/bin/bash

# Train classifier first
echo "Training classifiers for Ex2a: Binary Classification."
CUDA_VISIBLE_DEVICES=0 python benchmark.py --dataset All --Mixcase_filename "2llama_complete.json" --train_with_mixcase --log_name "Ex2a_binary_train" --ckpt_dir "./ckpt/Ex2a_binary" --seed 0 --mixcase_threshold 1 --train_threshold 10000 --finetune --mixcase_as_mgt

filenames=("2llama_complete.json" "2llama_rewrite.json" "2llama_polish_token.json" "2llama_polish_sentence.json" "gpt4_complete.json" "gpt4_rewrite.json" "gpt4_polish_token.json" "gpt4_polish_sentence.json" "gpt4_humanize.json" "MGT_polish_sentence.json" "MGT_polish_token.json" "2llama_humanize.json")

for filename in "${filenames[@]}"
do
    echo "Running benchmark for $filename."
    CUDA_VISIBLE_DEVICES=0 python benchmark.py --dataset All --Mixcase_filename "${filename}" --log_name "Ex2a_binary_${filename}l" --ckpt_dir "./ckpt/Ex2a_binary" --seed 0 --mixcase_threshold 1 --train_threshold 10000 --train_with_mixcase --finetune --mixcase_as_mgt --test_only
done

echo "All benchmarks completed."