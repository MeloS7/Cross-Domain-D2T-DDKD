# Released Data and Outputs

The lightweight release accompanies *Cross-Domain, Multi-Task Data-to-Text
Generation without In-Domain Training Data* (Findings of EMNLP 2026).
It supplies the paper's QUINTD-5 inputs and the saved artifacts needed to
inspect model outputs and analyze the released evaluations.

## Download and Extraction

The archive is approximately **5.7 MiB compressed** and **42.7 MiB extracted**,
with 388 files. All 110 generation files together occupy about **1.3 MiB
compressed** (10.3 MiB extracted).

From the repository root:

```bash
cd Cross-Domain-D2T-DDKD
unzip -q data.zip
```

`data_manifest.json` beside the archive records its checksum and each file's
relative path, size, SHA-256 checksum, and JSONL record count where applicable.
It is the complete inventory of the release.

## Data Layout

```text
data/
├── README.md                       # This data guide
├── SOURCES.md                      # Source attribution and terms
├── train/quintd5/{domain}/          # 500 real development inputs per domain
├── test/quintd/{domain}/            # 100 benchmark test inputs per domain
├── model_outputs/
│   ├── quintd/{domain}/
│   │   ├── outputs/                # Main-experiment generations
│   │   └── eval_res/               # Faithfulness and coverage judgments
│   └── quintd5/{domain}/
│       ├── outputs/                # Generations from the 500-input experiments
│       └── eval_res/               # Their GPT-5.1 judgments
├── human_eval/                     # Selected outputs and neutral annotator IDs
└── results/                        # Table 4 summary and source-file mapping
```

The five domains are `gsmarena`, `ice_hockey`, `owid`, `weather`, and `wikidata`.

## Contents

| Directory | What is included |
| --- | --- |
| `data/train/quintd5/{domain}/` | One `*_dev_500_chat_format.jsonl` per domain: 500 structured development inputs, 2,500 total. These are inference prompts with a fixed assistant prefix, not human references or teacher-generated training pairs. |
| `data/test/quintd/{domain}/` | Test tables and Qwen/Gemma chat prompts; 100 examples per domain. |
| `data/model_outputs/quintd/{domain}/outputs/` | All saved generations from the released main comparisons, including the system used in the human study. |
| `data/model_outputs/quintd/{domain}/eval_res/` | Saved faithfulness judgments and 40 coverage files: five domains × four systems × two judges. |
| `data/model_outputs/quintd5/{domain}/` | Ten generation files and ten GPT-5.1 annotation files for the 500-input comparisons, evaluated on the original 100 test inputs per domain. |
| `data/human_eval/` | 60 selected inputs and 240 system outputs; three annotation files containing 240 records each, saved selection indices, and annotation tools. |
| `data/results/` | `table4_error_counts.csv` and `table4_sources.json`, which identify the released annotations used to calculate the main comparison. |

All generations and evaluation annotations are retained. The outputs are small
relative to the omitted training data and support direct comparisons, output
length analysis, agreement analysis, and qualitative inspection.

## Record Formats

- **QUINTD-5 and test prompts:** JSONL with `messages`. The final assistant
  message in an input prompt is a fixed generation prefix, not a reference text.
- **Test tables:** JSONL with `formatted_input` and source fields.
- **Model generations:** JSONL with `response`.
- **Faithfulness judgments:** `table_idx` and `annotations`; error types are
  `0` incorrect fact, `1` not checkable, `2` misleading, and `3` other.
- **Coverage judgments:** `coverage_score`, `estimated_coverage_ratio`, and
  `parse_error`. Analysis excludes invalid records and reports valid counts.
- **Human annotations:** `annotator_id`, `instance_id`, `domain`, `table_idx`,
  `system`, `output_text`, and `errors`; `[SUM]` labels are analyzed separately.

Rows in input, generation, and evaluation files must remain aligned. Public
filenames use `sub` for subsampling and `pert` for perturbation; original
evaluation metadata can still use the legacy method labels.

## Human Annotation Privacy

The annotation files and their `annotator_id` fields use only `annotator_1`,
`annotator_2`, and `annotator_3`. The current archive provides no mapping to
personal identifiers. The records contain no annotator name, email, account, IP address,
or annotation timestamp fields. Dates, entity names, and locations in task
inputs or outputs describe the source data, not the annotators.

Renaming identifiers preserves the original selected examples, system outputs,
error spans, reasons, labels, and inter-annotator comparisons. The analysis
scripts use the same neutral numbering.

Earlier Git revisions retain superseded archives, including the former
letter-based identifiers. Updating the current package does not remove these
historical copies or shrink the repository's full Git history.

## Release Scope

WebNLG originals, prepared WebNLG SFT copies, bulk target-domain synthetic
training pairs, raw collection snapshots, routine preprocessing intermediates,
and trained weights are omitted. The source-initialization ablation is not
released as a complete artifact set.

The supplied artifacts support analysis of the main, QUINTD-5, human, and
coverage evaluations. Retraining requires obtaining or preparing training
data separately; the code examples accept user-supplied dataset paths.

See the [experiment guide](EXPERIMENTS.md) for analysis commands and
[data sources and attribution](DATA_SOURCES.md) for upstream resources and
their terms. The code's MIT license does not apply to the complete data archive.
