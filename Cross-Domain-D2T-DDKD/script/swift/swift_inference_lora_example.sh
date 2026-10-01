#!/usr/bin/env bash
set -Eeuo pipefail

# Generate with a trained LoRA adapter through ms-swift.
# Required: ADAPTERS, VAL_DATASET, RESULT_PATH, and either TASK_DESCRIPTION or RESPONSE_PREFIX.
: "${ADAPTERS:?Set ADAPTERS to a trained adapter directory}"
: "${VAL_DATASET:?Set VAL_DATASET to the chat-format JSONL input}"
: "${RESULT_PATH:?Set RESULT_PATH to the output JSONL path}"

if [[ -z "${RESPONSE_PREFIX:-}" ]]; then
  : "${TASK_DESCRIPTION:?Set TASK_DESCRIPTION or provide RESPONSE_PREFIX}"
  RESPONSE_PREFIX="<think>\n\n</think>\n\nSure! Here is the ${TASK_DESCRIPTION}:\n"
fi

swift_args=(
  infer
  --adapters "$ADAPTERS"
  --stream false
  --val_dataset "$VAL_DATASET"
  --temperature 0
  --max_new_tokens "${MAX_NEW_TOKENS:-4096}"
  --vllm_max_model_len "${MAX_MODEL_LEN:-13000}"
  --response_prefix "$RESPONSE_PREFIX"
  --infer_backend vllm
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
  --result_path "$RESULT_PATH"
  --write_batch_size "${WRITE_BATCH_SIZE:-1000}"
)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" swift "${swift_args[@]}"
