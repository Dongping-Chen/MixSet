#!/bin/bash

echo "Training classifiers for Ex3: Operation-wise Transfer Learning."
filenames=("2llama_complete.json" "2llama_rewrite.json" "2llama_polish_token.json" "2llama_polish_sentence.json" "gpt4_complete.json" "gpt4_rewrite.json" "gpt4_polish_token.json" "gpt4_polish_sentence.json" "gpt4_humanize.json" "MGT_polish_sentence.json" "MGT_polish_token.json" "2llama_humanize.json")

for filename in "${filenames[@]}"
do
    echo "Training classifiers for: $filename"
    CUDA_VISIBLE_DEVICES=1 python benchmark.py --dataset All --Mixcase_filename "$filename" --transfer_filename "2llama_rewrite.json" --train_with_mixcase --log_name "Ex3_operation_train_${filename}" --ckpt_dir "./ckpt/Ex3_operation_train_${filename1}" --seed 0 --mixcase_threshold 1 --train_threshold 1000 --finetune --only_supervised --mixcase_as_mgt
done

echo "All benchmarks completed."
