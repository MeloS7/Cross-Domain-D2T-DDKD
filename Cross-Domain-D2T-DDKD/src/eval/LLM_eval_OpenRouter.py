#!/usr/bin/env python3
"""Run LLM-as-a-judge evaluation through the OpenRouter-compatible API."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from openai import OpenAI
import yaml
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
DEFAULT_CONFIG = Path(__file__).with_name("eval_prompt_config.yaml")
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def load_jsonl_data(file_path: str | Path) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Prompt config must contain a YAML mapping.")
    if not isinstance(config.get("system_msg"), str) or not isinstance(
        config.get("prompt_template"), str
    ):
        raise ValueError("Prompt config must define string system_msg and prompt_template fields.")
    return config


def pre_process_data(
    data: list[dict[str, Any]],
    config: dict[str, Any],
    input_table: list[dict[str, Any]],
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
                    data=formatted_input, text=response
                ),
                "system_prompt": config["system_msg"],
                "original_response": response,
            }
        )
    return processed


def parse_judgment_response(content: str, original_text: str) -> dict[str, Any]:
    """Parse and validate a judge response before it can enter saved annotations."""
    json_content = content.strip()
    if json_content.startswith("```"):
        first_line, _, remainder = json_content.partition("\n")
        if first_line.strip().lower() not in {"```", "```json"}:
            raise ValueError("Judge response has an unsupported Markdown code fence.")
        json_content = remainder
        if json_content.rstrip().endswith("```"):
            json_content = json_content.rstrip()[:-3].rstrip()

    try:
        judgment = json.loads(json_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge response is not valid JSON: {exc.msg}.") from exc

    if not isinstance(judgment, dict) or not isinstance(judgment.get("errors"), list):
        raise ValueError("Judge response must be a JSON object with an 'errors' list.")

    for error_idx, error in enumerate(judgment["errors"]):
        if not isinstance(error, dict):
            raise ValueError(f"Annotation {error_idx} must be a JSON object.")
        reason = error.get("reason")
        span = error.get("text")
        category = error.get("type")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Annotation {error_idx} must have a non-empty string 'reason'.")
        if not isinstance(span, str) or not span:
            raise ValueError(f"Annotation {error_idx} must have a non-empty string 'text'.")
        if isinstance(category, bool) or not isinstance(category, int) or category not in {0, 1, 2, 3}:
            raise ValueError(f"Annotation {error_idx} has invalid type; expected an integer from 0 to 3.")
        if span.lower() not in original_text.lower():
            raise ValueError(f"Annotation {error_idx} span does not occur in the generated text.")

    return judgment


def create_annotation(
    text: str,
    judgment: dict[str, Any],
    table_idx: int,
    metric_name: str,
    dataset_name: str,
    model_name: str,
) -> dict[str, Any]:
    annotation_list = []
    current_pos = 0
    for raw_error in judgment["errors"]:
        error = dict(raw_error)
        span = error["text"]
        start_pos = text.lower().find(span.lower(), current_pos)
        if start_pos == -1:
            start_pos = text.lower().find(span.lower())
        if start_pos == -1:  # Also enforced by parse_judgment_response.
            raise ValueError(f"Annotation span at table_idx={table_idx} is not in the text.")
        error["start"] = start_pos
        annotation_list.append(error)
        current_pos = start_pos + len(span)

    return {
        "annotator_id": metric_name,
        "dataset": dataset_name,
        "model": model_name,
        "table_idx": table_idx,
        "annotations": annotation_list,
    }


def create_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY before running an API evaluation.")
    return OpenAI(
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
    )


def evaluate_ent_desc(
    data: list[dict[str, Any]],
    model: str,
    dataset_name: str,
    model_name: str,
    *,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    if client is None:
        client = create_client()

    annotations = []
    for idx, item in enumerate(tqdm(data, total=len(data), desc="Judge eval")):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["user_prompt"]},
                ],
                temperature=0,
                max_tokens=16384,
                seed=42,
            )
        except Exception as exc:
            # Do not echo SDK exception text: request errors may include sensitive headers.
            raise RuntimeError(
                f"API request failed for input row {idx} ({type(exc).__name__})."
            ) from None

        choices = getattr(completion, "choices", None)
        if not choices or not getattr(choices[0], "message", None):
            raise RuntimeError(f"API response for input row {idx} has no message choice.")
        content = choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"API response for input row {idx} has empty content.")

        try:
            judgment = parse_judgment_response(content, item["original_response"])
            annotation = create_annotation(
                item["original_response"],
                judgment,
                idx,
                model,
                dataset_name,
                model_name,
            )
        except ValueError as exc:
            raise RuntimeError(f"Invalid judge annotation for input row {idx}: {exc}") from exc
        annotations.append(annotation)

    return annotations


def save_annotations_jsonl(
    annotations: list[dict[str, Any]], output_file: str | Path
) -> None:
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
            for annotation in annotations:
                json.dump(annotation, file, ensure_ascii=False)
                file.write("\n")
        os.replace(temp_name, output_path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)
    logging.info("Saved %d annotations to %s", len(annotations), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-judge evaluation through OpenRouter."
    )
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("--input_table", required=True)
    parser.add_argument("-m", "--model", default="openai/gpt-5.1")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("-d", "--dataset_name", default="quintd")
    parser.add_argument("-n", "--model_name", default="gpt-5.1")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY before running an API evaluation.")

    data = load_jsonl_data(args.input_file)
    input_table = load_jsonl_data(args.input_table)
    config = load_yaml(args.config)
    data = pre_process_data(data, config, input_table)
    annotations = evaluate_ent_desc(
        data, args.model, args.dataset_name, args.model_name
    )
    save_annotations_jsonl(annotations, args.output_file)


if __name__ == "__main__":
    main()
