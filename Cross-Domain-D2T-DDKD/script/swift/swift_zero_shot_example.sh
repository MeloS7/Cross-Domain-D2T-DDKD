#!/usr/bin/env bash
set -Eeuo pipefail

# Zero-shot generation with a local model through ms-swift.
# Required: MODEL, VAL_DATASET, RESULT_PATH, and either TASK_DESCRIPTION or RESPONSE_PREFIX.
: "${MODEL:?Set MODEL to a local or Hugging Face model identifier}"
: "${VAL_DATASET:?Set VAL_DATASET to the chat-format JSONL input}"
: "${RESULT_PATH:?Set RESULT_PATH to the output JSONL path}"

if [[ -z "${RESPONSE_PREFIX:-}" ]]; then
  : "${TASK_DESCRIPTION:?Set TASK_DESCRIPTION or provide RESPONSE_PREFIX}"
  RESPONSE_PREFIX="<think>\n\n</think>\n\nSure! Here is the ${TASK_DESCRIPTION}:\n"
fi

swift_args=(
  infer
  --model "$MODEL"
  --stream false
  --infer_backend vllm
  --val_dataset "$VAL_DATASET"
  --max_new_tokens "${MAX_NEW_TOKENS:-4096}"
  --vllm_max_model_len "${MAX_MODEL_LEN:-13000}"
  --response_prefix "$RESPONSE_PREFIX"
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
  --result_path "$RESULT_PATH"
  --write_batch_size "${WRITE_BATCH_SIZE:-1000}"
  --temperature 0
  --logprobs False
  --seed "${SEED:-42}"
)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" swift "${swift_args[@]}"
