#!/usr/bin/env python3
"""
Compute average output length (in tokens and characters) for each system/model
on a given QUINTD-style dataset (Wikidata, weather, OWID, gsmarena, ice_hockey, ...).

This reuses the system discovery logic from `eval_judges_agreement.py` to find
all 14 systems per domain, and then loads the corresponding *_responses.jsonl
files to measure generation length.
"""

import argparse
import os
import statistics
import sys
from typing import Dict, List, Tuple

# Ensure the project src/ directory is on sys.path so we can import eval_judges_agreement
THIS_DIR = os.path.dirname(__file__)
SRC_ROOT = os.path.dirname(THIS_DIR)  # .../src
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from eval.eval_judges_agreement import SystemConfig, build_system_configs, load_jsonl


def count_tokens(text: str) -> int:
    """Simple whitespace tokenization."""
    if not text:
        return 0
    return len(text.split())


def pretty_model_name(cfg: SystemConfig) -> str:
    """Human-readable model name, consistent with eval_judges_agreement."""
    size = cfg.size.upper()
    if cfg.method == "zero_shot":
        return f"QWEN3-{size}"
    if cfg.method == "sft_lora":
        src = cfg.variant or "default"
        return f"QWEN3-{size}-LoRA-{src}"
    if cfg.method == "ddkd_zero_shot":
        var = (cfg.variant or "base").upper()
        return f"QWEN3-{size}-DDKD-ZS-{var}"
    if cfg.method == "ddkd_sft_lora":
        var = (cfg.variant or "base").upper()
        return f"QWEN3-{size}-DDKD-SFT-{var}"
    return cfg.name


def group_systems(systems: List[SystemConfig]) -> List[Tuple[str, List[SystemConfig]]]:
    """Group systems by method for nicer printing."""
    method_to_group_name = {
        "zero_shot": "Zero-Shot",
        "sft_lora": "SFT LoRA",
        "ddkd_zero_shot": "DDKD from Zero-Shot",
        "ddkd_sft_lora": "DDKD from SFT LoRA",
    }

    groups_map: Dict[str, List[SystemConfig]] = {}
    for cfg in systems:
        group_name = method_to_group_name.get(cfg.method, cfg.method)
        groups_map.setdefault(group_name, []).append(cfg)

    def sort_key(cfg: SystemConfig) -> Tuple[int, int, str]:
        size_order = {"1.7B": 0, "1.7b": 0, "8B": 1, "8b": 1, "32B": 2, "32b": 2}.get(
            cfg.size, 99
        )
        variant_order = {
            "": 0,
            "base": 0,
            "standard": 0,
            "aug": 1,
            "corr": 2,
            "mixed": 3,
        }.get(cfg.variant, 10)
        return (size_order, variant_order, cfg.variant)

    groups_list: List[Tuple[str, List[SystemConfig]]] = []
    for group_name, cfgs in groups_map.items():
        cfgs_sorted = sorted(cfgs, key=sort_key)
        groups_list.append((group_name, cfgs_sorted))

    return groups_list


def compute_length_stats(systems: List[SystemConfig]) -> Dict[str, Dict[str, float]]:
    """
    For each system, compute:
      - n_outputs
      - avg_tokens
      - std_tokens
      - avg_chars
      - std_chars
    Returns mapping: system_name -> stats dict.
    """
    results: Dict[str, Dict[str, float]] = {}
    for cfg in systems:
        if not cfg.outputs_path or not os.path.isfile(cfg.outputs_path):
            print(
                f"[Warning] Skipping system {cfg.name}: outputs file not found: "
                f"{cfg.outputs_path}"
            )
            continue

        outputs = load_jsonl(cfg.outputs_path)
        token_lengths: List[int] = []
        char_lengths: List[int] = []
        for rec in outputs:
            text = rec.get("response", "")
            token_lengths.append(count_tokens(text))
            char_lengths.append(len(text))

        if not token_lengths:
            continue

        avg_tok = statistics.mean(token_lengths)
        std_tok = statistics.pstdev(token_lengths) if len(token_lengths) > 1 else 0.0
        avg_char = statistics.mean(char_lengths)
        std_char = statistics.pstdev(char_lengths) if len(char_lengths) > 1 else 0.0

        results[cfg.name] = {
            "n_outputs": float(len(token_lengths)),
            "avg_tokens": avg_tok,
            "std_tokens": std_tok,
            "avg_chars": avg_char,
            "std_chars": std_char,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute average output length (tokens, characters) for all systems "
            "on one or more QUINTD datasets."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help=(
            "Base directory for paths (default: current directory). "
            "Should be the inner project root where 'data/' lives."
        ),
    )
    parser.add_argument(
        "--datasets",
        "--dataset",
        dest="datasets",
        type=str,
        required=True,
        help=(
            "Comma-separated list of datasets under data/model_outputs/quintd/, "
            "e.g. 'wikidata,weather,owid,gsmarena,ice_hockey'."
        ),
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not datasets:
        raise SystemExit("No valid datasets provided.")

    for dataset in datasets:
        print("\n" + "=" * 80)
        print(f"Dataset: {dataset}")
        print("=" * 80)

        systems = build_system_configs(args.base_dir, dataset)
        if not systems:
            print("  No systems found for this dataset, skipping.")
            continue

        length_stats = compute_length_stats(systems)
        if not length_stats:
            print("  No outputs with lengths found for this dataset, skipping.")
            continue

        groups = group_systems(systems)

        header = (
            f"{'Model':40s}"
            f"{'#Outputs':>10s}"
            f"{'AvgTokens':>12s}"
            f"{'StdTokens':>12s}"
            f"{'AvgChars':>12s}"
            f"{'StdChars':>12s}"
        )
        print(header)
        print("-" * len(header))

        for group_name, cfgs in groups:
            print(f"\n# {group_name}")
            for cfg in cfgs:
                stat = length_stats.get(cfg.name)
                if stat is None:
                    continue
                row = (
                    f"{pretty_model_name(cfg):40s}"
                    f"{int(stat['n_outputs']):10d}"
                    f"{stat['avg_tokens']:12.2f}"
                    f"{stat['std_tokens']:12.2f}"
                    f"{stat['avg_chars']:12.2f}"
                    f"{stat['std_chars']:12.2f}"
                )
                print(row)


if __name__ == "__main__":
    main()


