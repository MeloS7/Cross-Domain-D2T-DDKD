#!/usr/bin/env bash
set -Eeuo pipefail

# LoRA fine-tuning for original or augmented DDKD data.
# Required: MODE=base|augmented, MODEL, TRAIN_DATASET, OUTPUT_DIR, MODEL_NAME.
# VAL_DATASET is optional. Set MAX_LENGTH=8192 for shorter inputs and 13000 for long inputs.
: "${MODE:?Set MODE to either base or augmented}"
: "${MODEL:?Set MODEL to a local or Hugging Face model identifier}"
: "${TRAIN_DATASET:?Set TRAIN_DATASET to the training JSONL path}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for checkpoints and training outputs}"
: "${MODEL_NAME:?Set MODEL_NAME for the run metadata}"

case "$MODE" in
  base)
    swift_args=(
      sft
      --model "$MODEL"
      --train_type lora
      --dataset "$TRAIN_DATASET"
      --split_dataset_ratio "${SPLIT_DATASET_RATIO:-0.15}"
      --data_seed "${DATA_SEED:-42}"
      --dataset_shuffle True
      --loss_scale "${LOSS_SCALE:-last_round_with_ignore_empty_think}"
      --max_steps "${MAX_STEPS:-500}"
      --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-4}"
      --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-8}"
      --learning_rate "${LEARNING_RATE:-5e-5}"
      --lora_rank "${LORA_RANK:-8}"
      --lora_alpha "${LORA_ALPHA:-32}"
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
      --target_modules all-linear
      --eval_steps "${EVAL_STEPS:-5}"
      --output_dir "$OUTPUT_DIR"
      --save_steps "${SAVE_STEPS:-5}"
      --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
      --save_strategy steps
      --logging_steps "${LOGGING_STEPS:-10}"
      --max_length "${MAX_LENGTH:-13000}"
      --warmup_ratio "${WARMUP_RATIO:-0.05}"
      --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}"
      --model_name "$MODEL_NAME"
      --metric_for_best_model loss
      --greater_is_better False
      --early_stop_interval "${EARLY_STOP_INTERVAL:-5}"
    )
    ;;
  augmented)
    swift_args=(
      sft
      --model "$MODEL"
      --train_type lora
      --dataset "$TRAIN_DATASET"
      --data_seed "${DATA_SEED:-42}"
      --dataset_shuffle True
      --loss_scale "${LOSS_SCALE:-last_round_with_ignore_empty_think}"
      --num_train_epochs "${NUM_TRAIN_EPOCHS:-10}"
      --per_device_train_batch_size "${TRAIN_BATCH_SIZE:-8}"
      --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-16}"
      --learning_rate "${LEARNING_RATE:-1e-4}"
      --lora_rank "${LORA_RANK:-8}"
      --lora_alpha "${LORA_ALPHA:-32}"
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}"
      --target_modules all-linear
      --eval_steps "${EVAL_STEPS:-50}"
      --output_dir "$OUTPUT_DIR"
      --save_steps "${SAVE_STEPS:-50}"
      --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
      --save_strategy steps
      --logging_steps "${LOGGING_STEPS:-10}"
      --max_length "${MAX_LENGTH:-13000}"
      --warmup_ratio "${WARMUP_RATIO:-0.05}"
      --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}"
      --model_name "$MODEL_NAME"
      --metric_for_best_model loss
      --greater_is_better False
      --early_stop_interval "${EARLY_STOP_INTERVAL:-5}"
    )
    ;;
  *)
    printf 'Invalid MODE: choose base or augmented.\n' >&2
    exit 2
    ;;
esac

if [[ -n "${VAL_DATASET:-}" ]]; then
  swift_args+=(--val_dataset "$VAL_DATASET")
fi

swift "${swift_args[@]}"
