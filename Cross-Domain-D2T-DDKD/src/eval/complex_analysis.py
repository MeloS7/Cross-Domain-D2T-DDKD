#!/usr/bin/env python3
"""
Complexity analysis for QUINTD test sets (Gemma chat-format inputs).

For each dataset:
  - CSV-style given data (OWID):
      * count number of data rows (after the header line `date,value`)
      * report min / max / average rows per example

  - JSON-style given data (gsmarena, weather, ice_hockey):
      * parse the Python-dict-style JSON block
      * collect all distinct property names (dict keys) across the dataset
      * collect all distinct leaf values (as strings) across the dataset
      * also count distinct (property, value) pairs

  - Markdown-style given data (wikidata):
      * parse the bullet list:
            - property: value
        under the title and '---' separator
      * collect distinct properties, distinct values, and distinct (property, value) pairs

  - Prompt length (for all datasets):
      * using a Qwen tokenizer, compute token length of the *full* user prompt
      * report min / max / average token length per dataset

If `transformers` or the tokenizer is not available, the script will still
run the structural analyses and skip token-length statistics.
"""

import argparse
import ast
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_input_block_from_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Extract the given-data block from a list of chat messages.

    Looks for content between "Based on the given data:\\n```\\n" and "\\n```\\n\\n"
    in the first user message. Falls back to the full user content if markers are
    not found.
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        start_marker = "Based on the given data:\n```\n"
        end_marker = "\n```\n\n"
        start_idx = content.find(start_marker)
        if start_idx != -1:
            end_idx = content.find(end_marker, start_idx + len(start_marker))
            if end_idx != -1:
                return content[start_idx + len(start_marker) : end_idx].strip()
        # fallback: whole user message
        return content.strip()
    return ""


# ---------- CSV-style analysis (OWID) ----------


def analyze_csv_dataset(path: Path) -> None:
    """Compute row-count statistics for CSV-style OWID test set."""
    records = load_jsonl(path)
    row_counts: List[int] = []

    for rec in records:
        msgs = rec.get("messages", [])
        block = extract_input_block_from_messages(msgs)
        if not block:
            continue
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]

        # skip comment lines, find header "date,value"
        data_start = None
        for i, ln in enumerate(lines):
            if ln.strip().lower().startswith("date,value"):
                data_start = i + 1
                break
        if data_start is None:
            continue

        data_lines = [
            ln
            for ln in lines[data_start:]
            if not ln.strip().startswith("#") and ln.strip() != ""
        ]
        row_counts.append(len(data_lines))

    if not row_counts:
        print(f"[OWID CSV] No data rows found in {path}")
        return

    avg_rows = statistics.mean(row_counts)
    min_rows = min(row_counts)
    max_rows = max(row_counts)
    print("\n=== OWID CSV complexity (rows per example) ===")
    print(f"File: {path}")
    print(f"#examples: {len(row_counts)}")
    print(f"min rows: {min_rows}")
    print(f"max rows: {max_rows}")
    print(f"avg rows: {avg_rows:.2f}")


# ---------- JSON-style analysis (gsmarena/weather/ice_hockey) ----------


def _collect_json_props_vals(
    obj: Any,
    props: Set[str],
    vals: Set[str],
    pairs: Set[Tuple[str, str]],
    current_key: Optional[str] = None,
) -> None:
    """Recursively collect property names and leaf values from a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            props.add(str(k))
            _collect_json_props_vals(v, props, vals, pairs, current_key=str(k))
    elif isinstance(obj, list):
        for v in obj:
            _collect_json_props_vals(v, props, vals, pairs, current_key=current_key)
    else:
        # leaf value
        val_str = str(obj)
        vals.add(val_str)
        if current_key is not None:
            pairs.add((current_key, val_str))


def analyze_json_dataset(domain: str, path: Path) -> None:
    """Collect distinct properties / values / (prop, val) pairs from JSON-style inputs."""
    records = load_jsonl(path)
    props: Set[str] = set()
    vals: Set[str] = set()
    pairs: Set[Tuple[str, str]] = set()

    for rec in records:
        msgs = rec.get("messages", [])
        block = extract_input_block_from_messages(msgs)
        if not block:
            continue
        # JSON blocks are Python-literal dicts; use ast.literal_eval
        try:
            obj = ast.literal_eval(block)
        except Exception:
            continue
        _collect_json_props_vals(obj, props, vals, pairs)

    print(f"\n=== {domain} JSON complexity (properties / values) ===")
    print(f"File: {path}")
    print(f"#distinct properties: {len(props)}")
    print(f"#distinct values: {len(vals)}")
    print(f"#distinct (property, value) pairs: {len(pairs)}")


# ---------- Markdown-style analysis (wikidata) ----------


def analyze_markdown_dataset(path: Path) -> None:
    """Collect distinct properties / values / (prop, val) pairs from markdown inputs."""
    records = load_jsonl(path)
    props: Set[str] = set()
    vals: Set[str] = set()
    pairs: Set[Tuple[str, str]] = set()

    for rec in records:
        msgs = rec.get("messages", [])
        block = extract_input_block_from_messages(msgs)
        if not block:
            continue
        lines = [ln.rstrip() for ln in block.splitlines()]
        # find separator line '---'
        sep_idx = None
        for i, ln in enumerate(lines):
            if ln.strip() == "---":
                sep_idx = i
                break
        if sep_idx is None:
            continue
        for ln in lines[sep_idx + 1 :]:
            stripped = ln.strip()
            if not stripped.startswith("- "):
                continue
            item = stripped[2:]
            if ":" not in item:
                continue
            k, v = item.split(":", 1)
            prop = k.strip()
            val = v.strip()
            if not prop:
                continue
            props.add(prop)
            vals.add(val)
            pairs.add((prop, val))

    print("\n=== Wikidata markdown complexity (properties / values) ===")
    print(f"File: {path}")
    print(f"#distinct properties: {len(props)}")
    print(f"#distinct values: {len(vals)}")
    print(f"#distinct (property, value) pairs: {len(pairs)}")


# ---------- Qwen tokenizer-based prompt length analysis ----------


def maybe_load_qwen_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"[Warning] transformers not available ({e}); skip token-length analysis.")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir="./.cache")
    except Exception as e:  # pragma: no cover - depends on local env / cache
        print(
            f"[Warning] Failed to load tokenizer '{name}': {e}\n"
            "Token-length analysis will be skipped."
        )
        return None

    return tokenizer


def analyze_prompt_lengths(
    dataset_name: str,
    path: Path,
    tokenizer,
) -> None:
    """Compute min / max / mean prompt length (in Qwen tokens) for a dataset."""
    if tokenizer is None:
        return

    records = load_jsonl(path)
    lengths: List[int] = []

    for rec in records:
        msgs = rec.get("messages", [])
        user_text = ""
        for msg in msgs:
            if msg.get("role") == "user":
                user_text = str(msg.get("content", ""))
                break
        if not user_text:
            continue

        # Use tokenizer.encode to get a flat list of token IDs.
        try:
            token_ids = tokenizer.encode(user_text, add_special_tokens=False)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[Warning] Tokenization failed for one example in {dataset_name}: {e}")
            continue

        if not token_ids:
            continue
        lengths.append(len(token_ids))

    if not lengths:
        print(f"[Prompt] No prompts found for dataset {dataset_name} at {path}")
        return

    avg_len = statistics.mean(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    print(f"\n=== Prompt length (Qwen tokens) for {dataset_name} ===")
    print(f"File: {path}")
    print(f"#examples: {len(lengths)}")
    print(f"min tokens: {min_len}")
    print(f"max tokens: {max_len}")
    print(f"avg tokens: {avg_len:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze structural complexity and Qwen-token prompt length for QUINTD test sets."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/test/quintd",
        help="Base directory where domain subfolders (owid, weather, gsmarena, ...) live.",
    )
    parser.add_argument(
        "--qwen_tokenizer",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace tokenizer name for Qwen (used for prompt length stats).",
    )
    parser.add_argument(
        "--skip_tokenizer",
        action="store_true",
        help="If set, skip Qwen token-length analysis.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)

    # Dataset file paths (Gemma chat-format)
    owid_csv = base / "owid" / "owid_test_chat_format_gemma.jsonl"
    gsmarena_json = base / "gsmarena" / "gsmarena_test_chat_format_gemma.jsonl"
    weather_json = base / "weather" / "weather_test_chat_format_gemma.jsonl"
    ice_json = base / "ice_hockey" / "ice_hockey_test_chat_format_gemma.jsonl"
    wikidata_md = base / "wikidata" / "wikidata_test_chat_format_gemma.jsonl"

    # Run structural analyses
    if owid_csv.is_file():
        analyze_csv_dataset(owid_csv)
    else:
        print(f"[Warning] OWID CSV file not found: {owid_csv}")

    for domain, p in [
        ("gsmarena", gsmarena_json),
        ("weather", weather_json),
        ("ice_hockey", ice_json),
    ]:
        if p.is_file():
            analyze_json_dataset(domain, p)
        else:
            print(f"[Warning] JSON file for {domain} not found: {p}")

    if wikidata_md.is_file():
        analyze_markdown_dataset(wikidata_md)
    else:
        print(f"[Warning] Wikidata markdown file not found: {wikidata_md}")

    # Qwen tokenizer-based prompt lengths
    tokenizer = None
    if not args.skip_tokenizer:
        tokenizer = maybe_load_qwen_tokenizer(args.qwen_tokenizer)

    if tokenizer is not None:
        for name, p in [
            ("owid", owid_csv),
            ("gsmarena", gsmarena_json),
            ("weather", weather_json),
            ("ice_hockey", ice_json),
            ("wikidata", wikidata_md),
        ]:
            if p.is_file():
                analyze_prompt_lengths(name, p, tokenizer)


if __name__ == "__main__":
    main()


