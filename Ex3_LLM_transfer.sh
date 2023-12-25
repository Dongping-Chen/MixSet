#!/bin/bash

# Train classifier first without Mixset as Ex1 setting
echo "Training classifiers for Ex3: LLM-wise Transfer Learning without MixSet."
CUDA_VISIBLE_DEVICES=0 python benchmark.py --dataset All --Mixcase_filename "2llama_complete.json" --log_name "Ex3_LLM_train_without_mixset" --ckpt_dir "./ckpt/Ex3_LLM_train_without_mixset" --seed 0 --mixcase_threshold 1 --train_threshold 5000 --mixcase_as_mgt --MGT_only_GPT

filenames=("2llama_complete.json" "2llama_rewrite.json" "2llama_polish_token.json" "2llama_polish_sentence.json" "gpt4_complete.json" "gpt4_rewrite.json" "gpt4_polish_token.json" "gpt4_polish_sentence.json" "gpt4_humanize.json" "MGT_polish_sentence.json" "MGT_polish_token.json" "2llama_humanize.json")

for filename in "${filenames[@]}"
do
    echo "Running benchmark for $filename."
    CUDA_VISIBLE_DEVICES=0 python benchmark.py --dataset All --Mixcase_filename "${filename}" --log_name "Ex3_LLM_without_mixset_${filename}l" --ckpt_dir "./ckpt/Ex3_LLM_train_without_mixset" --seed 0 --mixcase_threshold 1 --train_threshold 5000 --mixcase_as_mgt --MGT_only_GPT --test_only
done

echo "All benchmarks completed."

# Train classifier first with Mixset as Ex2 setting
echo "Training classifiers for Ex3: LLM-wise Transfer Learning with MixSet."
CUDA_VISIBLE_DEVICES=0 python benchmark.py --dataset All --Mixcase_filename "2llama_complete.json" --log_name "Ex3_LLM_train_with_mixset" --ckpt_dir "./ckpt/Ex3_LLM_train_with_mixset" --seed 0 --mixcase_threshold 1 --train_threshold 5000 --train_with_mixcase --finetune --mixcase_as_mgt --MGT_only_GPT

filenames=("2llama_complete.json" "2llama_rewrite.json" "2llama_polish_token.json" "2llama_polish_sentence.json" "gpt4_complete.json" "gpt4_rewrite.json" "gpt4_polish_token.json" "gpt4_polish_sentence.json" "gpt4_humanize.json" "MGT_polish_sentence.json" "MGT_polish_token.json" "2llama_humanize.json")

for filename in "${filenames[@]}"
do
    echo "Running benchmark for $filename."
    CUDA_VISIBLE_DEVICES=0 python benchmark.py --dataset All --Mixcase_filename "${filename}" --log_name "Ex3_LLM_with_mixset_${filename}l" --ckpt_dir "./ckpt/Ex3_LLM_train_with_mixset" --seed 0 --mixcase_threshold 1 --train_threshold 5000 --train_with_mixcase --finetune --mixcase_as_mgt --MGT_only_GPT --test_only
done

echo "All benchmarks completed."