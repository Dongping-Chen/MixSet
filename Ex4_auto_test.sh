#!/bin/bash

# 定义变量数组
filenames=("2llama_complete.json" "2llama_rewrite.json" "2llama_polish_token.json" "2llama_polish_sentence.json" "gpt4_complete.json" "gpt4_rewrite.json" "gpt4_polish_token.json" "gpt4_polish_sentence.json" "gpt4_humanize.json" "MGT_polish_sentence.json" "MGT_polish_token.json" "2llama_humanize.json")
mixcase_thresholds=(0 0.5 1)
train_thresholds=(1000 4000 70000 10000)

# 遍历所有可能的组合
for filename in "${filenames[@]}"
do
    for mixcase_threshold in "${mixcase_thresholds[@]}"
    do
        for train_threshold in "${train_thresholds[@]}"
        do
            log_name="Ex4_mixcase_${mixcase_threshold}_train_${train_threshold}_seed_0"
            ckpt_dir="./ckpt/Ex4_mixcase_${mixcase_threshold}_train_${train_threshold}_seed_0"

            echo "Running benchmark with seed: $seed, mixcase threshold: $mixcase_threshold, train threshold: $train_threshold"
            CUDA_VISIBLE_DEVICES=1 python benchmark.py --dataset All --Mixcase_filename "$filename" --train_with_mixcase --log_name "$log_name" --ckpt_dir "$ckpt_dir" --seed 0 --mixcase_threshold "$mixcase_threshold" --train_threshold "$train_threshold" --test_only --finetune --mixcase_as_mgt
        done
    done
done

echo "All benchmarks completed."
