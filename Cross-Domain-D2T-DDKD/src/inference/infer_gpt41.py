#!/usr/bin/env python3
"""
Zero-shot generation on QUINTD test sets with GPT-4.1 via OpenRouter.

Input: chat-format jsonl (data/test/quintd/{domain}/{domain}_test_chat_format_gemma.jsonl)
Output: *_gpt4.1_responses.jsonl, each line:
  {"table_idx": int, "response": "<model output>"}

Notes:
- Environment variable OPENROUTER_API_KEY must be set.
- Performs generation only; no JSON parsing or evaluation.
"""

import argparse
import json
import os
import tempfile
import time
from typing import List, Dict, Any

from openai import OpenAI, OpenAIError
from tqdm import tqdm


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: List[dict], path: str) -> None:
    output_path = os.path.abspath(path)
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_name = f.name
            for row in rows:
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
        os.replace(temp_name, output_path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def create_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Environment variable OPENROUTER_API_KEY is not set.")
    return OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
    )


def call_chat(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    stream: bool = False,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    last_error_type = "unknown"
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                stream=stream,
            )
            if not stream:
                return resp.choices[0].message.content or ""
            # streaming: accumulate deltas
            chunks = []
            for chunk in resp:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    chunks.append(delta)
            return "".join(chunks)
        except OpenAIError as e:
            last_error_type = type(e).__name__
            wait = backoff * (attempt + 1)
            time.sleep(wait)
    raise RuntimeError(
        f"OpenRouter request failed after {retries} attempts ({last_error_type})."
    ) from None


def infer_file(
    input_path: str,
    output_path: str,
    model: str = "openai/gpt-4.1",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    seed: int = 42,
    stream: bool = False,
) -> None:
    data = load_jsonl(input_path)
    client = create_client()

    outputs = []
    for i, item in enumerate(tqdm(data, desc="infer", unit="ex")):
        messages = item.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError(f"Input row {i} must have a 'messages' list.")
        try:
            content = call_chat(
                client=client,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                stream=stream,
            )
            if stream:
                print(f"[stream] idx={i} => {content}")
        except Exception as e:
            raise RuntimeError(
                f"Generation failed for input row {i} ({type(e).__name__}); "
                "no output file was written."
            ) from None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                f"Generation returned empty content for input row {i}; "
                "no output file was written."
            )
        outputs.append({"table_idx": i, "response": content})

    save_jsonl(outputs, output_path)
    print(f"Saved {len(outputs)} responses to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GPT-4.1 via OpenRouter on QUINTD test chat-format JSONL."
    )
    parser.add_argument("--input_file", "-i", required=True, help="chat-format jsonl")
    parser.add_argument(
        "--output_file",
        "-o",
        help="output jsonl; default: replace suffix with _gpt4.1_responses.jsonl",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4.1",
        help="OpenRouter model name (default: openai/gpt-4.1)",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming responses (accumulate deltas to a single string).",
    )
    args = parser.parse_args()

    in_path = args.input_file
    out_path = args.output_file
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_gpt4.1_responses.jsonl"

    infer_file(
        input_path=in_path,
        output_path=out_path,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stream=args.stream,
    )


if __name__ == "__main__":
    main()
