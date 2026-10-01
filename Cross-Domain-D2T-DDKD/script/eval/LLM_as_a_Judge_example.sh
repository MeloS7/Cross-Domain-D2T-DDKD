#!/usr/bin/env bash
set -Eeuo pipefail

# Run faithfulness judging through OpenRouter. Results are written separately
# from the archived paper outputs.
# Required: INPUT_FILE, INPUT_TABLE, OUTPUT_FILE, MODEL, OPENROUTER_API_KEY.
: "${INPUT_FILE:?Set INPUT_FILE to the generated response JSONL}"
: "${INPUT_TABLE:?Set INPUT_TABLE to the matching formatted-input JSONL}"
: "${OUTPUT_FILE:?Set OUTPUT_FILE (for example, results/judge_output.jsonl)}"
: "${MODEL:?Set MODEL to an OpenRouter model identifier}"
: "${OPENROUTER_API_KEY:?Export OPENROUTER_API_KEY before evaluation}"

python src/eval/LLM_eval_OpenRouter.py \
  --input_file "$INPUT_FILE" \
  --input_table "$INPUT_TABLE" \
  --output_file "$OUTPUT_FILE" \
  --model "$MODEL" \
  --config "${CONFIG:-src/eval/eval_prompt_config.yaml}" \
  --dataset_name "${DATASET_NAME:-quintd}" \
  --model_name "${MODEL_NAME:-$MODEL}"
