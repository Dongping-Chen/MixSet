#!/bin/bash

# 定义文件名数组
filenames=("2llama_rewrite.json" "2llama_polish_token.json" "2llama_polish_sentence.json" "gpt4_complete.json" "gpt4_rewrite.json" "gpt4_polish_token.json" "gpt4_polish_sentence.json" "gpt4_humanize.json" "MGT_polish_sentence.json" "MGT_polish_token.json" "2llama_humanize.json")
gptzero_key=""
# 遍历文件名数组
for filename in "${filenames[@]}"
do
    echo "Running benchmark for $filename"
    python only_GPTzero.py --dataset All --Mixcase_filename "$filename" --log_name "GPTzero" --test_only --no_auc --gptzero_key "$gptzero_key"
done

echo "All benchmarks completed."
