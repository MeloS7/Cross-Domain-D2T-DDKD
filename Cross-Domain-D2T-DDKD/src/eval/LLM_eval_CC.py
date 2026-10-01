#!/usr/bin/env python3
"""Run LLM-as-a-judge Content Coverage evaluation through OpenRouter."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from openai import OpenAI
import yaml
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
DEFAULT_CONFIG = Path(__file__).with_name("eval_content_coverage.yaml")
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DOMAIN_DISPLAY = {
    "owid": "OWID",
    "gsmarena": "GSM Arena",
    "weather": "OpenWeather",
    "ice_hockey": "Ice Hockey",
    "wikidata": "Wikidata",
}


def load_jsonl_data(file_path: str | Path) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Prompt config must be a YAML mapping.")
    if not isinstance(config.get("system_msg"), str) or not isinstance(
        config.get("prompt_template"), str
    ):
        raise ValueError("Prompt config must define string system_msg and prompt_template fields.")
    return config


def pre_process_data(
    data: list[dict[str, Any]],
    config: dict[str, Any],
    input_table: list[dict[str, Any]],
    domain_display: str,
) -> list[dict[str, Any]]:
    if len(data) != len(input_table):
        raise ValueError(
            f"Response/table length mismatch: {len(data)} responses, "
            f"{len(input_table)} table rows."
        )

    processed = []
    for idx, (item, table) in enumerate(zip(data, input_table)):
        response = item.get("response")
        formatted_input = table.get("formatted_input")
        if not isinstance(response, str):
            raise ValueError(f"Response row {idx} must have a string 'response' field.")
        if not isinstance(formatted_input, str):
            raise ValueError(f"Input-table row {idx} must have a string 'formatted_input' field.")
        processed.append(
            {
                "user_prompt": config["prompt_template"].format(
                    DOMAIN=domain_display,
                    INPUT_DATA=formatted_input,
                    GEN_TEXT=response,
                ),
                "system_prompt": config["system_msg"],
                "original_response": response,
                "table_idx": item.get("table_idx", idx),
            }
        )
    return processed


def _extract_json_from_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        first_line, _, remainder = text.partition("\n")
        if first_line.strip().lower() not in {"```", "```json"}:
            raise ValueError("response has an unsupported Markdown code fence")
        text = remainder
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")

    score = parsed.get("coverage_score")
    ratio = parsed.get("estimated_coverage_ratio")
    missing = parsed.get("missing_points")
    notes = parsed.get("notes")
    if isinstance(score, bool) or not isinstance(score, int) or score not in {1, 2, 3, 4, 5}:
        raise ValueError("coverage_score must be an integer from 1 to 5")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError("estimated_coverage_ratio must be a number from 0 to 1")
    if not math.isfinite(float(ratio)) or not 0 <= ratio <= 1:
        raise ValueError("estimated_coverage_ratio must be a finite number from 0 to 1")
    if not isinstance(missing, list) or len(missing) > 3:
        raise ValueError("missing_points must be a list with at most three entries")
    for idx, point in enumerate(missing):
        if not isinstance(point, dict) or not isinstance(point.get("point"), str) or not point["point"].strip():
            raise ValueError(f"missing_points entry {idx} must have a non-empty point string")
        if point.get("status") not in {"ABSENT", "PARTIAL"}:
            raise ValueError(f"missing_points entry {idx} status must be ABSENT or PARTIAL")
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")
    return parsed


def _parse_error_result(item: dict[str, Any], idx: int, dataset_name: str, model_name: str,
                        judge_model: str, error: str) -> dict[str, Any]:
    return {
        "table_idx": item.get("table_idx", idx),
        "dataset": dataset_name,
        "model_name": model_name,
        "judge_model": judge_model,
        "coverage_score": None,
        "estimated_coverage_ratio": None,
        "missing_points": [],
        "notes": error,
        "parse_error": True,
    }


def create_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before running a coverage evaluation.")
    return OpenAI(
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
    )


def evaluate_content_coverage(
    data: list[dict[str, Any]],
    model: str,
    dataset_name: str,
    model_name: str,
    judge_model: str,
    *,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    if client is None:
        client = create_client()

    results = []
    for idx, item in enumerate(tqdm(data, total=len(data), desc="CC eval")):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["user_prompt"]},
                ],
                temperature=0,
                max_tokens=4096,
                seed=42,
            )
        except Exception as exc:
            # The parse_error flag is consumed by analyze_cc.py. Do not include
            # SDK response text here: an error object may contain sensitive data.
            detail = f"API request failed ({type(exc).__name__})"
            logging.error("Coverage evaluation failed for row %d (%s).", idx, type(exc).__name__)
            result = _parse_error_result(
                item, idx, dataset_name, model_name, judge_model, detail
            )
        else:
            try:
                content = completion.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("API returned empty content")
                parsed = _extract_json_from_response(content)
                result = {
                    "table_idx": item.get("table_idx", idx),
                    "dataset": dataset_name,
                    "model_name": model_name,
                    "judge_model": judge_model,
                    "coverage_score": parsed["coverage_score"],
                    "estimated_coverage_ratio": parsed["estimated_coverage_ratio"],
                    "missing_points": parsed["missing_points"],
                    "notes": parsed["notes"],
                    "parse_error": False,
                }
            except Exception as exc:
                detail = f"Invalid judge output ({type(exc).__name__})"
                if isinstance(exc, ValueError):
                    # These messages are generated locally by parser validation,
                    # and never include the raw model output.
                    detail = str(exc)
                logging.error("Coverage output validation failed for row %d: %s", idx, detail)
                result = _parse_error_result(
                    item, idx, dataset_name, model_name, judge_model, detail
                )
        results.append(result)
    return results


def save_results_jsonl(results: list[dict[str, Any]], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_name = file.name
            for result in results:
                json.dump(result, file, ensure_ascii=False)
                file.write("\n")
        os.replace(temp_name, output_path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    logging.info("Saved %d CC results to %s", len(results), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-judge Content Coverage evaluation via OpenRouter."
    )
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("--input_table", required=True)
    parser.add_argument("-m", "--model", default="openai/gpt-4o")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("-d", "--dataset_name", default="quintd")
    parser.add_argument("-n", "--model_name", default="gpt-4o")
    parser.add_argument("--domain", required=True, choices=sorted(DOMAIN_DISPLAY))
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running a coverage evaluation.")

    data = load_jsonl_data(args.input_file)
    input_table = load_jsonl_data(args.input_table)
    config = load_yaml(args.config)
    data = pre_process_data(data, config, input_table, DOMAIN_DISPLAY[args.domain])
    results = evaluate_content_coverage(
        data,
        args.model,
        args.dataset_name,
        args.model_name,
        judge_model=args.model,
    )
    save_results_jsonl(results, args.output_file)
    failures = sum(bool(result.get("parse_error")) for result in results)
    if failures:
        parser.exit(1, f"Coverage evaluation failed for {failures} of {len(results)} rows.\n")


if __name__ == "__main__":
    main()
