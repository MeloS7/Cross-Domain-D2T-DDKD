# Installation and Usage

This directory contains the data archive, evaluation code, inference code, and
local model examples. See the [project homepage](../README.md) for the framework
and key results, the [data guide](../docs/DATA.md) for archive contents, and the
[experiment guide](../docs/EXPERIMENTS.md) for the mapping to paper results.

## Setup

Run from the repository root:

```bash
cd Cross-Domain-D2T-DDKD
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.lock.txt
unzip -q data.zip
```

All commands below use this directory as the working directory. The analysis
dependency snapshot is tested with Python 3.12. Use `requirements.txt` for an
independently managed environment. GPU dependencies are installed separately.

## Analyze Saved Results

These commands use released files and do not need a GPU or API key:

```bash
python src/eval/analyze_judge_output.py \
  --input_file data/model_outputs/quintd/gsmarena/eval_res/zero_shot/gsmarena_zero_shot_gpt4.1_gpt-5.1.jsonl

python src/eval/summarize_table4.py --output results/table4_error_counts.csv
python src/eval/compute_norm.py --input results/table4_error_counts.csv

python src/eval/IAA_compute.py --base_dir data/human_eval
python src/eval/human_judge_agreement.py

python src/eval/eval_judges_agreement.py \
  --datasets wikidata,ice_hockey,weather,gsmarena,owid --no_verbose

python src/eval/analyze_cc.py --input_dir data/model_outputs/quintd
python src/eval/compute_cc_norm.py --data_dir data/model_outputs/quintd
python src/eval/inter_judge_agreement_cc.py --data_dir data/model_outputs/quintd
```

The first example reports **1.050 errors per output** and a **57.0% error rate**.
The human IAA script reports overall alpha values of **0.3692** (token),
**0.5784** (instance), **0.9023** (system), and **0.6926** (summary).

See the [experiment guide](../docs/EXPERIMENTS.md) for output lengths,
QUINTD-5 comparisons, and additional analyses.

## Generate and Evaluate New Text

The API scripts use an OpenRouter-compatible endpoint. Set credentials in your
shell:

```bash
export OPENROUTER_API_KEY='YOUR_API_KEY'
# Optional: export OPENROUTER_BASE_URL='https://openrouter.ai/api/v1'
mkdir -p results

python src/inference/infer_gpt41.py \
  --input_file data/test/quintd/gsmarena/gsmarena_test_chat_format.jsonl \
  --output_file results/gsmarena_gpt41_responses.jsonl \
  --model openai/gpt-4.1 --temperature 0 --max_tokens 2048 --seed 42

python src/eval/LLM_eval_OpenRouter.py \
  --input_file results/gsmarena_gpt41_responses.jsonl \
  --output_file results/gsmarena_gpt41_gpt51.jsonl \
  --input_table data/test/quintd/gsmarena/test_input_table.jsonl \
  --model openai/gpt-5.1 \
  --config src/eval/eval_prompt_config.yaml \
  --dataset_name quintd_gsmarena_test --model_name gpt-4.1
```

These commands make API requests. Tables and generations must have the same
row order and count. Failed API calls or invalid faithfulness judgments cause
a nonzero exit. Coverage failures retain `parse_error: true` and are excluded
from analysis.

The examples in `script/eval/` take paths and model IDs through environment
variables. Set `OUTPUT_FILE` to a new path under `results/`.

## Local Models and Training

The portable examples in `script/swift/` use
[MS-Swift](https://github.com/modelscope/ms-swift) and vLLM. Model, dataset,
adapter, and output locations are configured through environment variables.

The lightweight archive supplies test inputs and QUINTD-5 development inputs.
For training, obtain WebNLG from its upstream source or prepare your own
input–text pairs, then set `TRAIN_DATASET`. The archive omits WebNLG copies,
bulk synthetic training pairs, and checkpoints. A QUINTD-5 input file contains
an assistant response prefix only; it is not a supervised training dataset.

See [training and inference settings](../docs/EXPERIMENTS.md#training-and-inference)
for required variables, model-family differences, and the paper's settings.

## Local Verification

```bash
python -m unittest discover -s tests -v
```

Tests use local fixtures and stubs; they do not run models or contact an API.
