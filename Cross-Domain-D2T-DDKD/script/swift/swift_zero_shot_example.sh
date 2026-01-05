#! /bin/bash

# Zero-shot for quintd-weather
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model <path_to_model> \
    --stream  false\
    --infer_backend vllm \
    --val_dataset <path_to_val_dataset> \
    --max_new_tokens 4096 \
    --vllm_max_model_len 13000 \
    --response_prefix '<think>\n\n</think>\n\nSure! Here is the <task_description>:\n' \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_tensor_parallel_size 1 \
    --result_path <path_to_result_path> \
    --write_batch_size 1000 \
    --temperature 0 \
    --logprobs False \
    --seed 42