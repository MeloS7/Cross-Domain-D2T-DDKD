#!/bin/bash

# Note: for short-context datasets such as wikidata, ice_hockey, the max_length should be set to 8192; for long-context datasets such as owid, weather, gsmarena, the max_length should be set to 13000. 
# Otherwise, the model will run out of memory.

# Script for quintd-base sft with one GPU (small dataset)
swift sft \
    --model <path_to_model> \
    --train_type lora \
    --dataset <path_to_train_dataset> \
    --split_dataset_ratio 0.15 \
    --data_seed 42 \
    --dataset_shuffle True \
    --loss_scale 'last_round_with_ignore_empty_think' \
    --max_steps 500 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --target_modules all-linear \
    --eval_steps 5 \
    --output_dir .ckpt \
    --save_steps 5 \
    --save_total_limit 2 \
    --save_strategy steps \
    --logging_steps 10 \
    --max_length 13000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_name <path_to_model_name> \
    --metric_for_best_model loss \
    --greater_is_better False \
    --early_stop_interval 5

# Script for quintd-aug/pert/mixed sft with one GPU (augmented dataset)
swift sft \
    --model <path_to_model> \
    --train_type lora \
    --dataset <path_to_train_dataset> \
    --dataset_shuffle False \
    --val_dataset <path_to_val_dataset> \
    --data_seed 42 \
    --dataset_shuffle True \
    --loss_scale 'last_round_with_ignore_empty_think' \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --target_modules all-linear \
    --eval_steps 50 \
    --output_dir .ckpt \
    --save_steps 50 \
    --save_total_limit 2 \
    --save_strategy steps \
    --logging_steps 10 \
    --max_length 13000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_name <path_to_model_name> \
    --metric_for_best_model loss \
    --greater_is_better False \
    --early_stop_interval 5