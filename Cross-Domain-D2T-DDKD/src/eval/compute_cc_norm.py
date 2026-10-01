#!/usr/bin/env python3
"""Summarize released Content Coverage JSONL files and compute NormAvg."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:  # Support both package imports and direct script execution.
    from .analyze_cc import compute_stats
except ImportError:  # pragma: no cover - exercised by command-line use
    from analyze_cc import compute_stats


DOMAINS = ("gsmarena", "ice_hockey", "owid", "weather", "wikidata")
DOMAIN_LABELS = {
    "gsmarena": "GSM Arena",
    "ice_hockey": "Ice Hockey",
    "owid": "OWID",
    "weather": "OpenWeather",
    "wikidata": "Wikidata",
}
SYSTEMS = (
    "Qwen3-1.7B-ZS",
    "Qwen3-1.7B-SFT",
    "Qwen3-32B-SFT/ZS",
    "Qwen3-1.7B-DDKD-Best",
)
JUDGES = {"gpt5.1": "GPT-5.1", "gemini2.5-pro": "Gemini-2.5-Pro"}
DDKD_VARIANTS = {
    "gsmarena": "ddkd_zero_shot_qwen3_1.7b_pert",
    "weather": "ddkd_sft_lora_qwen3_1.7b_sub",
    "ice_hockey": "ddkd_sft_lora_qwen3_1.7b_mixed",
    "owid": "ddkd_sft_lora_qwen3_1.7b_mixed",
    "wikidata": "ddkd_sft_lora_qwen3_1.7b_mixed",
}


def normalize_judge(name: str) -> str | None:
    normalized = name.casefold().replace("-", "")
    if normalized == "gpt5.1":
        return "gpt5.1"
    if normalized == "gemini2.5pro":
        return "gemini2.5-pro"
    return None


def classify_system(domain: str, stem: str) -> str | None:
    normalized = stem.casefold()
    if not normalized.startswith(f"{domain}_"):
        return None
    if normalized.startswith(f"{domain}_{DDKD_VARIANTS[domain]}"):
        return SYSTEMS[3]
    if normalized.startswith(f"{domain}_zero_shot_qwen3_1.7b"):
        return SYSTEMS[0]
    if normalized.startswith(f"{domain}_sft_lora_") and "_qwen3_1.7b" in normalized:
        return SYSTEMS[1]
    if domain == "gsmarena":
        if normalized.startswith(f"{domain}_zero_shot_qwen3_32b"):
            return SYSTEMS[2]
    elif normalized.startswith(f"{domain}_sft_lora_") and "_qwen3_32b" in normalized:
        return SYSTEMS[2]
    return None


def discover_cc_files(input_dir: Path) -> dict[tuple[str, str, str], Path]:
    """Find the four paper-selected systems for each domain and both judges."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Coverage results directory does not exist: {input_dir}")

    found: dict[tuple[str, str, str], Path] = {}
    for path in sorted(input_dir.rglob("*_cc_*.jsonl")):
        stem = path.stem
        if "_cc_" not in stem:
            continue
        run_stem, judge_stem = stem.rsplit("_cc_", 1)
        judge = normalize_judge(judge_stem)
        if judge is None:
            continue
        domain = next(
            (candidate for candidate in DOMAINS if run_stem.casefold().startswith(f"{candidate}_")),
            None,
        )
        if domain is None:
            continue
        system = classify_system(domain, run_stem)
        if system is None:
            continue
        key = (domain, system, judge)
        if key in found:
            raise ValueError(
                f"Multiple coverage files match domain={domain}, system={system}, "
                f"judge={JUDGES[judge]}: {found[key].name} and {path.name}"
            )
        found[key] = path

    expected = {
        (domain, system, judge)
        for domain in DOMAINS
        for system in SYSTEMS
        for judge in JUDGES
    }
    missing = sorted(expected - found.keys())
    if missing:
        preview = ", ".join(
            f"{domain}/{system}/{JUDGES[judge]}" for domain, system, judge in missing
        )
        raise FileNotFoundError(f"Missing required coverage result files: {preview}")
    return found


def load_stats(path: Path) -> dict[str, Any]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path.name}, line {line_number}.") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {path.name}, line {line_number}.")
            records.append(record)
    if not records:
        raise ValueError(f"Coverage result file {path.name} is empty.")
    return compute_stats(records)


def minmax_normalize(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def summarize(input_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    files = discover_cc_files(input_dir)
    stats_by_key = {key: load_stats(path) for key, path in files.items()}

    # matrix[judge][metric][system][domain] = per-file mean
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for judge_key, judge_label in JUDGES.items():
        judge_summary: dict[str, dict[str, Any]] = {}
        for metric in ("score", "ratio"):
            stat_key = "mean_score" if metric == "score" else "mean_ratio"
            count_key = "n_valid" if metric == "score" else "n_valid_ratio"
            domain_values: dict[str, list[float]] = {domain: [] for domain in DOMAINS}
            metric_by_system: dict[str, dict[str, dict[str, Any]]] = {
                system: {} for system in SYSTEMS
            }
            for system in SYSTEMS:
                for domain in DOMAINS:
                    stats = stats_by_key[(domain, system, judge_key)]
                    value = stats[stat_key]
                    if value is None:
                        raise ValueError(
                            f"No valid {metric} values for {domain}/{system}/{judge_label}."
                        )
                    value = float(value)
                    domain_values[domain].append(value)
                    metric_by_system[system][domain] = {
                        "mean": value,
                        "n_valid": stats[count_key],
                        "n_parse_error": stats["n_parse_error"],
                        "n_total": stats["n_total"],
                    }

            normalized_by_domain = {
                domain: minmax_normalize(values)
                for domain, values in domain_values.items()
            }
            for system_idx, system in enumerate(SYSTEMS):
                normavg = statistics.mean(
                    normalized_by_domain[domain][system_idx] for domain in DOMAINS
                )
                for domain in DOMAINS:
                    metric_by_system[system][domain]["normalized"] = (
                        normalized_by_domain[domain][system_idx]
                    )
                    metric_by_system[system][domain]["NormAvg"] = normavg
            judge_summary[metric] = metric_by_system
        summary[judge_key] = judge_summary
    return summary


def print_summary(summary: dict[str, dict[str, dict[str, Any]]]) -> None:
    for judge_key, judge_label in JUDGES.items():
        for metric, metric_label in (("score", "coverage score"), ("ratio", "coverage ratio")):
            rows = summary[judge_key][metric]
            headers = ["System"] + [DOMAIN_LABELS[d] for d in DOMAINS] + ["NormAvg"]
            values = []
            for system in SYSTEMS:
                values.append(
                    [
                        system,
                        *[
                            f"{rows[system][domain]['mean']:.3f} "
                            f"(n={rows[system][domain]['n_valid']}, "
                            f"err={rows[system][domain]['n_parse_error']})"
                            for domain in DOMAINS
                        ],
                        f"{rows[system][DOMAINS[0]]['NormAvg']:.3f}",
                    ]
                )
            widths = [
                max(len(headers[idx]), *(len(row[idx]) for row in values))
                for idx in range(len(headers))
            ]
            separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
            print(f"\n{judge_label}: mean {metric_label} and per-domain NormAvg")
            print(separator)
            print("|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, widths)) + "|")
            print(separator)
            for row in values:
                print("|" + "|".join(f" {v:<{w}} " for v, w in zip(row, widths)) + "|")
            print(separator)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CC per-domain means and NormAvg from released JSONL files."
    )
    parser.add_argument(
        "--input_dir",
        "--data_dir",
        dest="input_dir",
        required=True,
        help="Directory containing domain folders with *_cc_<judge>.jsonl files.",
    )
    args = parser.parse_args()
    print_summary(summarize(Path(args.input_dir)))


if __name__ == "__main__":
    main()
