# !/bin/bash

# Eval with GPT-5.1
python src/eval/LLM_eval_OpenRouter.py \
-i data/model_outputs/quintd/gsmarena/outputs/zero_shot/gsmarena_test_chat_format_gpt4.1_responses.jsonl \
-o data/model_outputs/quintd/gsmarena/eval_res/zero_shot/test_gsmarena_chat_format_gpt4.1_gpt-5.1.jsonl \
--input_table data/test/quintd/gsmarena/test_input_table.jsonl \
-m openai/gpt-5.1 \
-c src/eval/eval_prompt_config.yaml \
-d quintd_gsmarena_test \
-n openai/gpt-5.1 \

# Eval with Gemini 2.5-pro
python src/eval/LLM_eval_OpenRouter.py \
-i data/model_outputs/quintd/gsmarena/outputs/ddkd_zero_shot/gsmarena_test_gemma3_1b_it_ZS_mixed_responses.jsonl \
-o data/model_outputs/quintd/gsmarena/eval_res/ddkd_zero_shot/test_gsmarena_gemma3_1b_it_ZS_mixed_gemini2.5-pro.jsonl \
--input_table data/test/quintd/gsmarena/test_input_table.jsonl \
-m google/gemini-2.5-pro \
-c src/eval/eval_prompt_config.yaml \
-d quintd_gsmarena_test \
-n google/gemini-2.5-pro \