#!/bin/bash

# 定义变量数组
seeds=(0)
mixcase_thresholds=(0 0.5 1)
train_thresholds=(1000 4000 7000 10000)

# 遍历所有可能的组合
for seed in "${seeds[@]}"
do
    for mixcase_threshold in "${mixcase_thresholds[@]}"
    do
        for train_threshold in "${train_thresholds[@]}"
        do
            log_name="Ex4_mixcase_${mixcase_threshold}_train_${train_threshold}_seed_${seed}"
            ckpt_dir="./ckpt/Ex4_mixcase_${mixcase_threshold}_train_${train_threshold}_seed_${seed}"

            echo "Running benchmark with seed: $seed, mixcase threshold: $mixcase_threshold, train threshold: $train_threshold"
            CUDA_VISIBLE_DEVICES=0 python benchmark.py --dataset All --Mixcase_filename "2llama_complete.json" --train_with_mixcase --log_name "$log_name" --ckpt_dir "$ckpt_dir" --seed "$seed" --mixcase_threshold "$mixcase_threshold" --train_threshold "$train_threshold" --finetune --mixcase_as_mgt
        done
    done
done

echo "All benchmarks completed."
