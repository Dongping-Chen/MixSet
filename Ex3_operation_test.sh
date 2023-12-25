#!/bin/bash

echo "Evaluating classifiers for Ex3: Operation-wise Transfer Learning."
filenames1=("2llama_complete.json" "2llama_rewrite.json" "2llama_polish_token.json" "2llama_polish_sentence.json" "gpt4_complete.json" "gpt4_rewrite.json" "gpt4_polish_token.json" "gpt4_polish_sentence.json" "gpt4_humanize.json" "MGT_polish_sentence.json" "MGT_polish_token.json" "2llama_humanize.json")
filenames2=("2llama_complete.json" "2llama_rewrite.json" "2llama_polish_token.json" "2llama_polish_sentence.json" "gpt4_complete.json" "gpt4_rewrite.json" "gpt4_polish_token.json" "gpt4_polish_sentence.json" "gpt4_humanize.json" "MGT_polish_sentence.json" "MGT_polish_token.json" "2llama_humanize.json")

for filename1 in "${filenames1[@]}"
do
    for filename2 in "${filenames2[@]}"
    do
        echo "Evaluating classifiers trained from $filename1 and transfer to $filename2."
        CUDA_VISIBLE_DEVICES=0 python benchmark.py --dataset All --Mixcase_filename "$filename1" --transfer_filename "$filename2" --train_with_mixcase --log_name "Ex3_operation_${filename1}_${filename2}l" --ckpt_dir "./ckpt/Ex3_operation_train_${filename1}" --seed 0 --mixcase_threshold 1 --train_threshold 1000 --test_only --only_supervised --finetune --mixcase_as_mgt
    done
done

echo "All benchmarks completed."
