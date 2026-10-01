#!/usr/bin/env python3
"""
Analyze Content Coverage (CC) evaluation results from LLM_eval_CC.py output.

Computes mean coverage_score and estimated_coverage_ratio per file,
and prints a summary table. Optionally saves to CSV.

Usage:
  # Single file
  python src/data_analysis/analyze_cc.py -i data/model_outputs/quintd/wikidata/eval_res/sft_lora/wikidata_sft_lora_cc_gpt5.1.jsonl

  # Multiple files
  python src/data_analysis/analyze_cc.py -i file1.jsonl file2.jsonl file3.jsonl

  # Scan directory for *_cc_*.jsonl
  python src/data_analysis/analyze_cc.py -d data/model_outputs/quintd/wikidata/eval_res

  # Save table to CSV
  python src/data_analysis/analyze_cc.py -d data/model_outputs/quintd/wikidata/eval_res -o cc_summary.csv
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import List, Optional


def load_cc_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compute_stats(records: List[dict]) -> dict:
    """Compute mean coverage_score and estimated_coverage_ratio, excluding parse_error rows."""
    valid = [r for r in records if not r.get("parse_error") and r.get("coverage_score") is not None]
    ratio_valid = [r for r in valid if r.get("estimated_coverage_ratio") is not None]

    n_total = len(records)
    n_valid = len(valid)
    n_parse_error = sum(1 for r in records if r.get("parse_error"))
    n_ratio_valid = len(ratio_valid)

    if not valid:
        return {
            "n_total": n_total,
            "n_valid": n_valid,
            "n_valid_ratio": n_ratio_valid,
            "n_parse_error": n_parse_error,
            "mean_score": None,
            "mean_ratio": None,
            "std_score": None,
            "std_ratio": None,
        }

    scores = [r["coverage_score"] for r in valid]
    ratios = [r["estimated_coverage_ratio"] for r in ratio_valid] if ratio_valid else []

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_valid_ratio": n_ratio_valid,
        "n_parse_error": n_parse_error,
        "mean_score": statistics.mean(scores),
        "mean_ratio": statistics.mean(ratios) if ratios else None,
        "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "std_ratio": statistics.stdev(ratios) if len(ratios) > 1 else (0.0 if ratios else None),
    }


def short_name(path: Path) -> str:
    """Extract a short run name from file path (e.g. sft_lora, zero_shot)."""
    stem = path.stem
    # Remove _cc_* suffix (judge model name)
    if "_cc_" in stem:
        stem = stem.split("_cc_")[0]
    # Prefer parent dir + stem for uniqueness
    parts = path.parts
    if "eval_res" in parts:
        idx = parts.index("eval_res")
        if idx + 1 < len(parts):
            sub = parts[idx + 1]  # e.g. sft_lora, zero_shot
            return f"{sub}/{stem}" if sub != stem else stem
    return stem


def extract_judge_from_path(path: Path) -> str:
    """Extract judge model from filename: *_{judge}.jsonl -> part after last _ before .jsonl.
    For *_cc_{judge}.jsonl, the judge is the segment after _cc_."""
    stem = path.stem
    if "_cc_" in stem:
        return stem.split("_cc_", 1)[1]
    # Fallback: last segment after final underscore
    if "_" in stem:
        return stem.rsplit("_", 1)[1]
    return stem


def format_table(rows: List[dict], col_widths: Optional[dict] = None) -> str:
    """Format rows as an ASCII table."""
    if not rows:
        return "No data."
    keys = list(rows[0].keys())
    if col_widths is None:
        col_widths = {k: max(len(str(k)), 4) for k in keys}
    for r in rows:
        for k in keys:
            v = r.get(k, "")
            col_widths[k] = max(col_widths.get(k, 4), len(str(v)), len(str(k)))

    lines = []
    sep = "+" + "+".join("-" * (col_widths[k] + 2) for k in keys) + "+"
    header = "|" + "|".join(f" {str(k):<{col_widths[k]}} " for k in keys) + "|"
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for r in rows:
        line = "|" + "|".join(
            f" {_fmt(r.get(k)):<{col_widths[k]}} " for k in keys
        ) + "|"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze CC evaluation results: mean coverage_score and estimated_coverage_ratio."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        default=[],
        help="Input JSONL file(s) from LLM_eval_CC.py",
    )
    parser.add_argument(
        "-d",
        "--input_dir",
        type=str,
        default=None,
        help="Scan directory recursively for *_cc_*.jsonl files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Save table to CSV file",
    )
    args = parser.parse_args()

    paths: List[Path] = []
    if args.input:
        paths = [Path(p) for p in args.input]
    if args.input_dir:
        root = Path(args.input_dir)
        paths.extend(sorted(root.rglob("*_cc_*.jsonl")))
    paths = list(dict.fromkeys(paths))  # dedup preserving order

    if not paths:
        print("No input files. Use -i file.jsonl or -d directory.")
        return

    rows = []
    for p in paths:
        if not p.exists():
            print(f"[Warning] File not found: {p}")
            continue
        records = load_cc_jsonl(p)
        stats = compute_stats(records)
        # Judge: from record if present (new format), else parse from filename
        judge = ""
        if records and records[0].get("judge_model"):
            judge = records[0]["judge_model"]
        else:
            judge = extract_judge_from_path(p)
        rows.append({
            "run": short_name(p),
            "judge": judge,
            "mean_score": stats["mean_score"],
            "mean_ratio": stats["mean_ratio"],
            "std_score": stats["std_score"],
            "std_ratio": stats["std_ratio"],
            "n": stats["n_valid"],
            "n_err": stats["n_parse_error"],
        })

    # Print table (run, judge, mean_score, mean_ratio, n, n_err)
    display_rows = []
    for r in rows:
        std_s = f"±{r['std_score']:.2f}" if r["std_score"] is not None else "-"
        std_r = f"±{r['std_ratio']:.2f}" if r["std_ratio"] is not None else "-"
        display_rows.append({
            "run": r["run"],
            "judge": r["judge"],
            "mean_score": r["mean_score"],
            "mean_ratio": r["mean_ratio"],
            "std_score": std_s,
            "std_ratio": std_r,
            "n": r["n"],
            "n_err": r["n_err"],
        })
    print("\nContent Coverage Summary")
    print("=" * 60)
    print(format_table(display_rows))
    print()

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as f:
            f.write("run,judge,mean_score,mean_ratio,std_score,std_ratio,n,n_parse_error\n")
            for r in rows:
                ms = f"{r['mean_score']:.3f}" if r["mean_score"] is not None else ""
                mr = f"{r['mean_ratio']:.3f}" if r["mean_ratio"] is not None else ""
                ss = f"{r['std_score']:.3f}" if r["std_score"] is not None else ""
                sr = f"{r['std_ratio']:.3f}" if r["std_ratio"] is not None else ""
                f.write(f"{r['run']},{r['judge']},{ms},{mr},{ss},{sr},{r['n']},{r['n_err']}\n")
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
