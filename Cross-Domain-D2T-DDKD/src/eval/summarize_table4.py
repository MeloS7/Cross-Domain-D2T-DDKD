#!/usr/bin/env python3
"""Recompute Table 4 error-count means from published judge annotations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, TextIO

try:  # Support both package imports and direct script execution.
    from .path_utils import resolve_case_insensitive_path
except ImportError:  # pragma: no cover - exercised by command-line use
    from path_utils import resolve_case_insensitive_path


DOMAINS = ("wikidata", "ice_hockey", "weather", "gsmarena", "owid")
DOMAIN_COLUMNS = {
    "wikidata": "Wikidata",
    "ice_hockey": "Ice Hockey",
    "weather": "OpenWeather",
    "gsmarena": "GSM Arena",
    "owid": "OWID",
}
ERROR_TYPES = {0, 1, 2, 3}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
        raise ValueError(f"Annotation file {path.name} is empty.")
    return records


def mean_error_count(path: Path) -> float:
    records = load_jsonl(path)
    total = 0
    for row_number, record in enumerate(records, 1):
        annotations = record.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(
                f"Expected an annotations list in {path.name}, record {row_number}."
            )
        for annotation in annotations:
            if not isinstance(annotation, dict):
                raise ValueError(
                    f"Expected an annotation object in {path.name}, record {row_number}."
                )
            error_type = annotation.get("type")
            if (
                isinstance(error_type, bool)
                or not isinstance(error_type, int)
                or error_type not in ERROR_TYPES
            ):
                raise ValueError(
                    f"Invalid annotation type in {path.name}, record {row_number}; "
                    "expected an integer from 0 to 3."
                )
            total += 1
    return total / len(records)


def resolve_base_path(base_dir: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("Source entries must be relative to --base_dir.")
    requested = base_dir / candidate
    resolved = resolve_case_insensitive_path(requested)
    if resolved is None:
        raise FileNotFoundError(f"A listed annotation file is missing: {candidate.as_posix()}")
    return Path(resolved)


def build_rows(base_dir: Path, sources_path: Path) -> list[dict[str, str]]:
    with sources_path.open("r", encoding="utf-8") as file:
        sources = json.load(file)
    if not isinstance(sources, dict) or not sources:
        raise ValueError("The source manifest must be a non-empty JSON object.")

    rows: list[dict[str, str]] = []
    for system, domain_paths in sources.items():
        if not isinstance(system, str) or not isinstance(domain_paths, dict):
            raise ValueError("Each source-manifest entry must map a system name to domain paths.")
        if set(domain_paths) != set(DOMAINS):
            raise ValueError(
                f"System {system!r} must list exactly these domains: {', '.join(DOMAINS)}."
            )
        row = {"System": system}
        for domain in DOMAINS:
            relative_path = domain_paths[domain]
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"System {system!r} has an invalid {domain} path.")
            path = resolve_base_path(base_dir, relative_path)
            row[DOMAIN_COLUMNS[domain]] = f"{mean_error_count(path):.2f}"
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], output: TextIO) -> None:
    writer = csv.DictWriter(
        output,
        fieldnames=["System", *(DOMAIN_COLUMNS[domain] for domain in DOMAINS)],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute per-system mean error counts from Table 4 source annotations."
    )
    parser.add_argument("--base_dir", default=".", help="Project directory containing data/.")
    parser.add_argument(
        "--sources",
        default="data/results/table4_sources.json",
        help="JSON manifest of per-system, per-domain annotation paths.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="CSV output path (default: stdout).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    sources_path = Path(args.sources)
    if not sources_path.is_absolute():
        sources_path = base_dir / sources_path
    rows = build_rows(base_dir, sources_path)
    if args.output == "-":
        write_csv(rows, sys.stdout)
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = base_dir / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as file:
            write_csv(rows, file)


if __name__ == "__main__":
    main()
