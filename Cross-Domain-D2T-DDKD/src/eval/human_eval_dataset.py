#!/usr/bin/env python3
"""
Construct the unified human-evaluation dataset (inputs + 4 system outputs)
for QUINTD benchmarks using pre-selected indices.

Design:
  - Inputs:
      1) indices JSON: data/human_eval/human_eval_indices.json
         Structure: { domain: [table_idx, ...], ... }, exactly 12 indices per domain
      2) Test inputs per domain (chat format):
         data/test/quintd/{domain}/{domain}_test_chat_format.jsonl
      3) Four systems' model outputs per domain:
         data/model_outputs/quintd/{domain}/outputs/{method}/*.jsonl
         (Qwen3-32B & Qwen3-1.7B zero-shot / SFT / DDKD, etc.)

  - For each domain and each selected table_idx:
      * Extract the input data block from chat_format JSONL
        (from "Based on the given data:\\n```\\n" to "\\n```\\n\\n", inclusive);
      * Fetch the corresponding response from the four *_responses.jsonl files.

  - Output:
      data/human_eval/human_eval_dataset.json
      Each record looks like:
        {
          "domain": "owid",
          "table_idx": 7,
          "input_block": "<raw input data block>",
          "outputs": {
            "primary_32B": "<best model output>",
            "zero_shot_1.7B": "<qwen3-1.7B zero-shot>",
            "sft_lora_1.7B": "<qwen3-1.7B SFT LoRA>",
            "ddkd_best": "<best DDKD model output>"
          }
        }
"""

import argparse
import json
import os
from typing import Dict, List


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_input_block_from_messages(messages: List[dict]) -> str:
    """
    Find the message containing the table text and extract the content between
    "Based on the given data:\\n```\\n" and "\\n```\\n\\n".
    If markers are missing, fall back to the first user message.
    """
    start_token = "Based on the given data:\n```\n"
    end_token = "\n```\n\n"

    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        if start_token in content:
            start = content.index(start_token)
            end_pos = content.find(end_token, start + len(start_token))
            if end_pos == -1:
                # If end marker is missing, take from start to end of text
                return content[start:]
            else:
                return content[start : end_pos + len(end_token)]

    # Fallback: return first user message if markers not found
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
    # Final fallback: concatenate all message contents
    return "\n\n".join(str(m.get("content", "")) for m in messages)


def build_system_output_paths(base_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Provide output paths for four systems per domain:
      - primary_32B      : best model (32B, zero-shot or SFT LoRA depending on domain)
      - zero_shot_1.7B   : Qwen3-1.7B zero-shot
      - sft_lora_1.7B    : Qwen3-1.7B SFT LoRA
      - ddkd_best        : selected best DDKD variant
    """

    def p(rel: str) -> str:
        return os.path.join(base_dir, rel)

    cfg: Dict[str, Dict[str, Dict[str, str]]] = {}

    # OWID: primary = SFT LoRA 32B (on_webnlg_csv)
    cfg["owid"] = {
        "primary_32B": {
            "path": p(
                "data/model_outputs/quintd/owid/outputs/sft_lora/owid_sft_lora_on_webnlg_csv_qwen3_32b_responses.jsonl"
            )
        },
        "zero_shot_1.7B": {
            "path": p(
                "data/model_outputs/quintd/owid/outputs/zero_shot/owid_zero_shot_qwen3_1.7b_responses.jsonl"
            )
        },
        "sft_lora_1.7B": {
            "path": p(
                "data/model_outputs/quintd/owid/outputs/sft_lora/owid_sft_lora_on_webnlg_csv_qwen3_1.7b_responses.jsonl"
            )
        },
        "ddkd_best": {
            "path": p(
                "data/model_outputs/quintd/owid/outputs/ddkd_sft_lora/owid_ddkd_sft_lora_qwen3_1.7b_mixed_responses.jsonl"
            )
        },
    }

    # gsmarena: primary = zero_shot 32B
    cfg["gsmarena"] = {
        "primary_32B": {
            "path": p(
                "data/model_outputs/quintd/gsmarena/outputs/zero_shot/gsmarena_zero_shot_qwen3_32b_responses.jsonl"
            )
        },
        "zero_shot_1.7B": {
            "path": p(
                "data/model_outputs/quintd/gsmarena/outputs/zero_shot/gsmarena_zero_shot_qwen3_1.7b_responses.jsonl"
            )
        },
        "sft_lora_1.7B": {
            "path": p(
                "data/model_outputs/quintd/gsmarena/outputs/sft_lora/gsmarena_sft_lora_on_webnlg_json_qwen3_1.7b_responses.jsonl"
            )
        },
        "ddkd_best": {
            "path": p(
                "data/model_outputs/quintd/gsmarena/outputs/ddkd_zero_shot/gsmarena_ddkd_zero_shot_qwen3_1.7b_mixed_responses.jsonl"
            )
        },
    }

    # weather: primary = SFT LoRA 32B (on_webnlg_json)
    cfg["weather"] = {
        "primary_32B": {
            "path": p(
                "data/model_outputs/quintd/weather/outputs/sft_lora/weather_sft_lora_on_webnlg_json_qwen3_32B_responses.jsonl"
            )
        },
        "zero_shot_1.7B": {
            "path": p(
                "data/model_outputs/quintd/weather/outputs/zero_shot/weather_zero_shot_qwen3_1.7B_responses.jsonl"
            )
        },
        "sft_lora_1.7B": {
            "path": p(
                "data/model_outputs/quintd/weather/outputs/sft_lora/weather_sft_lora_on_webnlg_json_qwen3_1.7B_responses.jsonl"
            )
        },
        "ddkd_best": {
            "path": p(
                "data/model_outputs/quintd/weather/outputs/ddkd_sft_lora/weather_ddkd_sft_lora_qwen3_1.7b_sub_responses.jsonl"
            )
        },
    }

    # wikidata: primary = SFT LoRA 32B (on_webnlg_mkd)
    cfg["wikidata"] = {
        "primary_32B": {
            "path": p(
                "data/model_outputs/quintd/wikidata/outputs/sft_lora/wikidata_sft_lora_on_webnlg_mkd_qwen3_32B_responses.jsonl"
            )
        },
        "zero_shot_1.7B": {
            "path": p(
                "data/model_outputs/quintd/wikidata/outputs/zero_shot/wikidata_zero_shot_qwen3_1.7B_responses.jsonl"
            )
        },
        "sft_lora_1.7B": {
            "path": p(
                "data/model_outputs/quintd/wikidata/outputs/sft_lora/wikidata_sft_lora_on_webnlg_mkd_qwen3_1.7B_responses.jsonl"
            )
        },
        # Corresponds to eval: wikidata_ddkd_sft_lora_qwen3_32B_mixed_gpt-5.1.jsonl
        "ddkd_best": {
            "path": p(
                "data/model_outputs/quintd/wikidata/outputs/ddkd_sft_lora/wikidata_qwen3_32B_webnlg_mkd_mixed_responses.jsonl"
            )
        },
    }

    # ice_hockey: primary = SFT LoRA 32B (on_webnlg_json)
    cfg["ice_hockey"] = {
        "primary_32B": {
            "path": p(
                "data/model_outputs/quintd/ice_hockey/outputs/sft_lora/ice_hockey_sft_lora_on_webnlg_json_qwen3_32B_responses.jsonl"
            )
        },
        "zero_shot_1.7B": {
            "path": p(
                "data/model_outputs/quintd/ice_hockey/outputs/zero_shot/ice_hockey_zero_shot_qwen3_1.7B_responses.jsonl"
            )
        },
        "sft_lora_1.7B": {
            "path": p(
                "data/model_outputs/quintd/ice_hockey/outputs/sft_lora/ice_hockey_sft_lora_on_webnlg_json_qwen3_1.7B_responses.jsonl"
            )
        },
        "ddkd_best": {
            "path": p(
                "data/model_outputs/quintd/ice_hockey/outputs/ddkd_sft_lora/ice_hockey_ddkd_sft_lora_qwen3_1.7b_mixed_responses.jsonl"
            )
        },
    }

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble human evaluation dataset (inputs + 4 system outputs) "
            "for QUINTD domains using pre-selected indices."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Base directory where 'data/' lives (default: current directory).",
    )
    parser.add_argument(
        "--indices_path",
        type=str,
        default="data/human_eval/human_eval_indices.json",
        help="Path to JSON with selected indices per domain.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/human_eval/human_eval_dataset.json",
        help="Path to save the assembled human evaluation dataset (JSON list).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    indices_path = args.indices_path
    output_file = args.output_file

    if not os.path.isabs(indices_path):
        indices_path = os.path.join(base_dir, indices_path)
    if not os.path.isabs(output_file):
        output_file = os.path.join(base_dir, output_file)

    with open(indices_path, "r", encoding="utf-8") as f:
        indices_per_domain: Dict[str, List[int]] = json.load(f)

    system_paths = build_system_output_paths(base_dir)

    all_records: List[dict] = []

    for domain, indices in indices_per_domain.items():
        indices_sorted = sorted(indices)

        # Load inputs (chat format)
        input_path = os.path.join(
            base_dir,
            "data",
            "test",
            "quintd",
            domain,
            f"{domain}_test_chat_format.jsonl",
        )
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input chat_format file not found: {input_path}")
        input_records = load_jsonl(input_path)

        # Load outputs for 4 systems
        if domain not in system_paths:
            raise ValueError(f"No system output paths configured for domain={domain}")

        sys_cfg = system_paths[domain]
        # Check paths and load
        outputs_by_system: Dict[str, List[dict]] = {}
        for sys_name, info in sys_cfg.items():
            path = info["path"]
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Output file for system '{sys_name}' in domain '{domain}' "
                    f"not found: {path}"
                )
            outputs_by_system[sys_name] = load_jsonl(path)

        # Basic consistency check
        n_inputs = len(input_records)
        for sys_name, outs in outputs_by_system.items():
            if len(outs) != n_inputs:
                raise ValueError(
                    f"Length mismatch in domain={domain}, system={sys_name}: "
                    f"inputs={n_inputs}, outputs={len(outs)}"
                )

        # Assemble human-eval samples by index
        for idx in indices_sorted:
            if idx < 0 or idx >= n_inputs:
                raise IndexError(
                    f"Index {idx} out of range for domain={domain} (n_inputs={n_inputs})"
                )

            inp_rec = input_records[idx]
            messages = inp_rec.get("messages", [])
            input_block = extract_input_block_from_messages(messages)

            outputs_entry: Dict[str, str] = {}
            for sys_name, outs in outputs_by_system.items():
                out_rec = outs[idx]
                # Convention: response field stores the generated text
                text = out_rec.get("response", "")
                outputs_entry[sys_name] = text

            all_records.append(
                {
                    "domain": domain,
                    "table_idx": idx,
                    "input_block": input_block,
                    "outputs": outputs_entry,
                }
            )

    # Save to JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"Saved human evaluation dataset with {len(all_records)} entries to: {output_file}")


if __name__ == "__main__":
    main()


