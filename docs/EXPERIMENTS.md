# Experiment guide

Commands assume that `data.zip` has been extracted and that the working
directory is `Cross-Domain-D2T-DDKD/`. Analysis scripts accept `--base_dir` or
`--data_dir` where applicable, so data can also be stored in another location.
New outputs should be written to `results/`.

## Paper results and artifacts

| Paper result | Released artifacts | Analysis |
| --- | --- | --- |
| Main faithfulness comparison (Table 4 and per-domain appendix tables) | `data/model_outputs/quintd/*/eval_res/`; 500-input results in `data/model_outputs/quintd5/*/eval_res/` | `summarize_table4.py`, `analyze_judge_output.py`, `compute_norm.py` |
| QUINTD-5 construction (Appendix D) | `data/train/quintd5/*/*_dev_500_chat_format.jsonl`; 500 chat examples per domain | Prepared input files; teacher examples are in the adjacent ZS/SFT subdirectories |
| LLM faithfulness agreement (Appendix H) | Paired GPT-5.1/Gemini annotations and their generated texts under `data/model_outputs/quintd/` | `eval_judges_agreement.py` |
| Human evaluation and agreement (Appendix I, including Table 12) | `data/human_eval/` | `IAA_compute.py`, `human_judge_agreement.py` |
| Output-length analysis | `data/model_outputs/quintd/*/outputs/` | `eval_output_length.py` |
| Coverage scores and normalized comparison (Appendix K, Tables 64–65) | 40 `*_cc_*.jsonl` files: five domains × four systems × two judges | `analyze_cc.py`, `compute_cc_norm.py` |
| Coverage inter-judge agreement (Tables 16–17) | The same paired coverage annotations | `inter_judge_agreement_cc.py` |

The source-initialization ablation (Appendix L) is not included as a complete
artifact set in this release. Full raw collection snapshots, routine conversion
intermediates, and trained weights are also omitted. These omissions do not
prevent analysis of the released main, QUINTD-5, human, and coverage results.

## Main table and QUINTD-5

`data/results/table4_error_counts.csv` contains mean error counts computed from
the saved GPT-5.1 annotations. `table4_sources.json` identifies the annotation
file used for each system and domain, using public relative paths only.

```bash
mkdir -p results
python src/eval/summarize_table4.py --output results/table4_error_counts.csv
python src/eval/compute_norm.py --input results/table4_error_counts.csv

# One 500-input experiment: saved outputs were evaluated on 100 test inputs.
python src/eval/analyze_judge_output.py \
  --input_file data/model_outputs/quintd5/gsmarena/eval_res/gsmarena_qwen3_1.7B_lora_rank8_alpha32_quintd5_SFT_gpt-5.1.jsonl
```

NormAvg for error counts uses domain-wise min–max normalization, averaged across
domains; lower is better. These values are calculated from the released
annotations rather than hardcoded paper numbers.

QUINTD-5 provides five input files and ten teacher-generated training files.
Each file has 500 records. Teacher training files contain the full `messages`
conversation, including the generated assistant response; duplicated inference
metadata is omitted. Original model responses and evaluation annotations are
preserved in the corresponding results directories.

## Faithfulness, human evaluation, and lengths

```bash
python src/eval/eval_judges_agreement.py \
  --datasets wikidata,ice_hockey,weather,gsmarena,owid --no_verbose

python src/eval/human_judge_agreement.py
python src/eval/IAA_compute.py --base_dir data/human_eval

python src/eval/eval_output_length.py \
  --datasets wikidata,ice_hockey,weather,gsmarena,owid

# Rebuild the selected inputs and outputs without replacing the original file.
python src/eval/human_eval_dataset.py \
  --output_file results/human_eval_dataset.json

# Structural analysis without downloading a tokenizer.
python src/eval/complex_analysis.py --skip_tokenizer
```

Faithfulness agreement discovers 14 paired Qwen systems per domain. It excludes
coverage files and the separate human-study Wikidata 32B distilled system.
The human study retains its original system selection, including that system;
its previously missing generations and judge annotations are supplied.

`eval_output_length.py` measures whitespace-separated words and characters.
Qwen-token prompt length analysis in `complex_analysis.py` additionally requires
Transformers and access to the chosen tokenizer via `--qwen_tokenizer`.

Use the saved `human_eval_indices.json` for analyses of the published human
annotations. `create_human_eval_indices.py` can generate a new stratified
selection; generating a new selection does not create corresponding human
annotations.

## Content coverage

```bash
python src/eval/analyze_cc.py \
  --input_dir data/model_outputs/quintd --output results/coverage_summary.csv
python src/eval/compute_cc_norm.py --data_dir data/model_outputs/quintd
python src/eval/inter_judge_agreement_cc.py --data_dir data/model_outputs/quintd
```

The coverage comparison includes four systems in every domain: Qwen3-1.7B ZS,
Qwen3-1.7B WebNLG SFT, the selected large teacher, and the selected distilled
student. The large teacher uses ZS for GSM Arena and WebNLG SFT for the other
domains. Selected students use ZS/Pert for GSM Arena, SFT/Sub for Weather, and
SFT/Mixed for the other domains.

Coverage Score is ordinal (1–5); Estimated Coverage Ratio is continuous (0–1).
Invalid original judgments are excluded and valid record counts are reported.
NormAvg uses per-domain min–max normalized mean coverage scores, averaged
across domains; higher is better. Coverage systems retain the original coverage
experiment selection, which can differ from the selected human-study system.

The released filenames use `sub` and `pert`; legacy `aug`/`corr` labels can
remain inside the saved evaluation metadata. No judgments are regenerated
during analysis.

## Training and inference

The scripts under `script/swift/` are portable MS-Swift examples, parameterized
through environment variables. They do not require a particular cluster or
job scheduler. Appendix A of the paper describes the experiment settings.

| Script | Configuration |
| --- | --- |
| `swift_sft_on_webnlg_example.sh` | Model, training/development data, model name, and output directory |
| `swift_sft_distill_example.sh` | Model, training data, model name, output directory; `MODE=base` or `MODE=augmented`; optional validation data |
| `swift_zero_shot_example.sh` | Model, validation inputs, and result path |
| `swift_inference_lora_example.sh` | Adapter location, validation inputs, and result path |

All paths are supplied by the person running the experiment. Model identifiers
such as `Qwen/Qwen3-1.7B` or `google/gemma-3-1b-it` may be used in place of local
model paths. Set `MAX_LENGTH`/context settings and batch sizes to suit the
domain and available GPU memory.

Qwen examples use the reasoning-aware loss scale and response prefix from the
supplied templates. For Gemma, set `LOSS_SCALE=last_round` and provide an
appropriate `RESPONSE_PREFIX` without Qwen's thinking tokens. Use the released
Gemma-prepared SFT examples when training Gemma on WebNLG. Consult the scripts
for their complete environment-variable defaults and required variables.

The analysis lockfile records the Python 3.12 CPU environment used to verify
this release. It does not reconstruct the historical MS-Swift, PyTorch,
Transformers, or vLLM training environment. The GPU examples were checked with
a command stub; GPU training and paid API evaluation were not rerun.

## New API evaluations

Set `OPENROUTER_API_KEY` in the environment. Optionally set
`OPENROUTER_BASE_URL` for another compatible endpoint. No key or local
configuration file is distributed.

`script/eval/LLM_as_a_Judge_example.sh` and
`script/eval/LLM_content_coverage_example.sh` demonstrate new evaluations.
Set `INPUT_FILE`, `INPUT_TABLE`, `OUTPUT_FILE`, and `MODEL`; the coverage example
also requires `DOMAIN`. Set `OUTPUT_FILE` to a new file under `results/`.
Inputs and tables must have the same count and row order. Faithfulness output
is written after the whole batch succeeds. Coverage failures are saved with
`parse_error: true`, and the command exits unsuccessfully when failures occur.
Original saved coverage failures remain marked and are handled by the analysis
scripts.

## Local verification

```bash
python -m unittest discover -s tests -v
```

The tests exercise filename resolution, evaluation failure handling, and shell
argument construction without external API calls or model execution.
