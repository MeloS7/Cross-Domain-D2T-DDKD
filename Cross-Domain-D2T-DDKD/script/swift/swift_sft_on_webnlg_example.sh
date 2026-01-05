#!/bin/bash

# Script for sft on webnlg_csv/markdown/json with one GPU
srun swift sft \
    --model <path_to_model> \
    --train_type lora \
    --dataset <path_to_train_dataset> \
    --dataset_shuffle True \
    --val_dataset <path_to_val_dataset> \
    --loss_scale 'last_round' \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --target_modules all-linear \
    --eval_steps 100 \
    --output_dir .ckpt \
    --save_steps 100 \
    --save_total_limit 2 \
    --save_strategy steps \
    --logging_steps 5 \
    --max_length 13000 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_name <path_to_model_name> \
    --metric_for_best_model loss \
    --greater_is_better False \
    --early_stop_interval 5