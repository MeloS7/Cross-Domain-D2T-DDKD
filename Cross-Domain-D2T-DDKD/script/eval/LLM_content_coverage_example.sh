#!/usr/bin/env bash
set -Eeuo pipefail

# Run Content Coverage judging through OpenRouter.
# Required: INPUT_FILE, INPUT_TABLE, OUTPUT_FILE, MODEL, DOMAIN, OPENROUTER_API_KEY.
: "${INPUT_FILE:?Set INPUT_FILE to the generated response JSONL}"
: "${INPUT_TABLE:?Set INPUT_TABLE to the matching formatted-input JSONL}"
: "${OUTPUT_FILE:?Set OUTPUT_FILE (for example, results/coverage_output.jsonl)}"
: "${MODEL:?Set MODEL to an OpenRouter model identifier}"
: "${DOMAIN:?Set DOMAIN to wikidata, ice_hockey, weather, gsmarena, or owid}"
: "${OPENROUTER_API_KEY:?Export OPENROUTER_API_KEY before evaluation}"

python src/eval/LLM_eval_CC.py \
  --input_file "$INPUT_FILE" \
  --input_table "$INPUT_TABLE" \
  --output_file "$OUTPUT_FILE" \
  --model "$MODEL" \
  --domain "$DOMAIN" \
  --config "${CONFIG:-src/eval/eval_content_coverage.yaml}" \
  --dataset_name "${DATASET_NAME:-quintd}" \
  --model_name "${MODEL_NAME:-$MODEL}"
