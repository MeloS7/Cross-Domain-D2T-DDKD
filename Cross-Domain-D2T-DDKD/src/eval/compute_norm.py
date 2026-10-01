#!/usr/bin/env python3
"""Compute NormAvg for a CSV or TSV table of per-domain values."""

from __future__ import annotations

import argparse
import csv
from typing import Any


def load_table(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as file:
        sample = file.read(2048)
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        delimiter = "\t" if "\t" in sample and "," not in first_line else ","
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter=delimiter))


def compute_normavg(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    if not rows:
        return []

    domains = []
    for key, value in rows[0].items():
        if key.lower() in {"system", "normavg"} or value is None or not str(value).strip():
            continue
        domains.append(key)

    minmax: dict[str, tuple[float, float]] = {}
    for domain in domains:
        values = []
        for row in rows:
            try:
                values.append(float(row[domain]))
            except (KeyError, TypeError, ValueError):
                continue
        if values:
            minmax[domain] = min(values), max(values)

    output = []
    for row in rows:
        normalized = []
        for domain in domains:
            try:
                value = float(row[domain])
            except (KeyError, TypeError, ValueError):
                continue
            low, high = minmax.get(domain, (0.0, 0.0))
            normalized.append(0.0 if high <= low else (value - low) / (high - low))
        output.append(
            {
                "System": row.get("System", ""),
                "NormAvg": sum(normalized) / len(normalized) if normalized else 0.0,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute NormAvg for a per-domain table.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV or TSV table.")
    args = parser.parse_args()
    results = compute_normavg(load_table(args.input))
    print("System,NormAvg")
    for row in results:
        print(f"{row['System']},{row['NormAvg']:.4f}")


if __name__ == "__main__":
    main()
