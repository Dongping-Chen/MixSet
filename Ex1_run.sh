#!/bin/bash

# Train classifier first
echo "Training classifiers for Ex1."
CUDA_VISIBLE_DEVICES=0 python benchmark.py --dataset All --Mixcase_filename "2llama_complete.json" --log_name "Ex1_train" --ckpt_dir "./ckpt/Ex1" --seed 0 --train_threshold 10000 --mixcase_as_mgt --no_auc

filenames=("2llama_complete.json" "2llama_rewrite.json" "2llama_polish_token.json" "2llama_polish_sentence.json" "gpt4_complete.json" "gpt4_rewrite.json" "gpt4_polish_token.json" "gpt4_polish_sentence.json" "gpt4_humanize.json" "MGT_polish_sentence.json" "MGT_polish_token.json" "2llama_humanize.json")

for filename in "${filenames[@]}"
do
    echo "Running benchmark for $filename."
    CUDA_VISIBLE_DEVICES=0 python benchmark.py --dataset All --Mixcase_filename "$filename" --log_name "Ex1_${filename}l" --ckpt_dir "./ckpt/Ex1" --seed 0 --train_threshold 10000 --mixcase_as_mgt --test_only --no_auc
done

echo "All benchmarks completed."
