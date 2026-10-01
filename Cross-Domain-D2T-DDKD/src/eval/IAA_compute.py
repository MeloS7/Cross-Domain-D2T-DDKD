#!/usr/bin/env python3
"""
Compute multi-annotator IAA for annotators 1, 2, and 3 using Krippendorff's Alpha.

We follow the expert suggestion and keep summarization as nominal (no ordinal order):
  - Token level: nominal alpha over token labels {0=no error, 1/2/3=error type}, [SUM] skipped.
  - Instance level: interval alpha over error counts per example (excluding [SUM]).
  - System level: interval alpha over per-system average error counts (excluding [SUM]).
  - Summarization level: nominal alpha over summary labels (short/medium/none) from [SUM].

"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import simpledorff

ANNOTATORS = ["1", "2", "3"]
SUMMARY_LABELS = {"none", "short", "medium"}
# Token labels: 0 = no error, 1 = type0, 2 = type1, 3 = type2, 4 = type3
# (so type0 is not conflated with "no error")


def load_data(base_dir: Path) -> Dict[str, Dict[str, dict]]:
    """
    Load human_annotations_annotator_1.jsonl, _2.jsonl, and _3.jsonl.
    Return: {instance_id: {annotator: record}}, only keeping instances annotated by all.
    """
    merged: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for ann in ANNOTATORS:
        path = base_dir / f"human_annotations_annotator_{ann}.jsonl"
        if not path.exists():
            print(f"[Warn] Missing file: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                inst_id = rec.get("instance_id")
                if inst_id:
                    merged[inst_id][ann] = rec

    valid = {k: v for k, v in merged.items() if len(v) == len(ANNOTATORS)}
    print(f"Loaded {len(valid)} instances with full annotations.")
    return valid


def tokenize(text: str) -> List[Dict[str, int]]:
    tokens = []
    for m in re.finditer(r"\S+", text):
        tokens.append({"start": m.start(), "end": m.end()})
    return tokens


def token_labels(text: str, errors: List[dict]) -> List[int]:
    """
    Map errors to tokens.
    0 = no error; 1 = type0; 2 = type1; 3 = type2; 4 = type3.
    If multiple errors overlap a token, keep the first encountered (order in JSONL).
    [SUM] entries are ignored.
    """
    toks = tokenize(text)
    labels = [0] * len(toks)
    if not toks:
        return labels

    for err in errors:
        if isinstance(err.get("text", ""), str) and err.get("text", "").strip().lower() == "[sum]":
            continue
        etype = err.get("type")
        if etype not in (0, 1, 2, 3):
            continue
        label_val = etype + 1  # shift by 1 so type0 != no-error
        start = err.get("start", -1)
        span_text = str(err.get("text", "")).strip()
        if start is None or start < 0:
            start = text.find(span_text)
        if start < 0:
            continue
        end = start + len(span_text)
        for i, t in enumerate(toks):
            if t["start"] < end and t["end"] > start and labels[i] == 0:
                labels[i] = label_val
    return labels


def norm_summary_label(errors: List[dict]) -> str:
    for e in errors:
        if isinstance(e.get("text", ""), str) and e.get("text", "").strip().lower() == "[sum]":
            lab = str(e.get("summary", "none")).strip().lower() or "none"
            if lab not in SUMMARY_LABELS:
                lab = "none"
            return lab
    return "none"


def main():
    parser = argparse.ArgumentParser(
        description="Compute Krippendorff's alpha at token / instance-count / summary levels."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/human_eval",
        help=(
            "Directory containing human_annotations_annotator_1.jsonl, _2.jsonl, "
            "and _3.jsonl"
        ),
    )
    args = parser.parse_args()

    data = load_data(Path(args.base_dir))

    token_rows = []  # multiclass 0/1/2/3/4
    count_rows = []  # all types (interval)
    summary_rows = []
    system_rows = []  # all types (interval)

    # Per-type containers
    token_rows_by_type = {t: [] for t in (0, 1, 2, 3)}
    count_rows_by_type = {t: [] for t in (0, 1, 2, 3)}
    system_rows_by_type = {t: [] for t in (0, 1, 2, 3)}

    for inst_id, ann_dict in data.items():
        # assume output_text identical across annotators; take one as reference
        ref_text = next(iter(ann_dict.values())).get("output_text", "")
        # parse system name from instance_id (domain|table_idx|system)
        parts = inst_id.split("|")
        system_name = parts[2] if len(parts) >= 3 else "unknown_system"

        for ann, rec in ann_dict.items():
            errs = rec.get("errors", [])

            # Token level
            labels = token_labels(ref_text, errs)
            for i, lab in enumerate(labels):
                token_rows.append({"unit_id": f"{inst_id}_{i}", "annotator": ann, "class": lab})
            # Per-type token (binary)
            for t in (0, 1, 2, 3):
                for i, lab in enumerate(labels):
                    token_rows_by_type[t].append(
                        {
                            "unit_id": f"{inst_id}_{i}",
                            "annotator": ann,
                            "class": 1 if lab == (t + 1) else 0,
                        }
                    )

            # Instance-level count (excluding [SUM])
            real_errs = [
                e for e in errs
                if not (isinstance(e.get("text", ""), str) and e.get("text", "").strip().lower() == "[sum]")
            ]
            count_rows.append({"unit_id": inst_id, "annotator": ann, "value": len(real_errs)})
            for t in (0, 1, 2, 3):
                cnt_t = sum(1 for e in real_errs if e.get("type") == t)
                count_rows_by_type[t].append({"unit_id": inst_id, "annotator": ann, "value": cnt_t})

            # Summary label (from [SUM])
            summary_rows.append({"unit_id": inst_id, "annotator": ann, "value": norm_summary_label(errs)})

        # System-level: average errors per system per annotator
        # (we accumulate and compute after the loop)
        # prepare to aggregate per annotator
        if "sys_agg" not in locals():
            sys_agg = defaultdict(lambda: defaultdict(list))  # sys_agg[annotator][system] = list of counts
            sys_agg_t = {t: defaultdict(lambda: defaultdict(list)) for t in (0, 1, 2, 3)}
        for ann, rec in ann_dict.items():
            errs = [
                e for e in rec.get("errors", [])
                if not (isinstance(e.get("text", ""), str) and e.get("text", "").strip().lower() == "[sum]")
            ]
            cnt = len(errs)
            sys_agg[ann][system_name].append(cnt)
            for t in (0, 1, 2, 3):
                cnt_t = sum(1 for e in errs if e.get("type") == t)
                sys_agg_t[t][ann][system_name].append(cnt_t)

    # build system_rows after aggregation
    for ann, sys_dict in sys_agg.items():
        for sys_name, cnts in sys_dict.items():
            if not cnts:
                continue
            avg = sum(cnts) / len(cnts)
            system_rows.append({"unit_id": sys_name, "annotator": ann, "value": avg})
    for t in (0, 1, 2, 3):
        for ann, sys_dict in sys_agg_t[t].items():
            for sys_name, cnts in sys_dict.items():
                if not cnts:
                    continue
                avg = sum(cnts) / len(cnts)
                system_rows_by_type[t].append({"unit_id": sys_name, "annotator": ann, "value": avg})

    # Compute alphas
    print("\n=== Token-level Krippendorff's alpha (nominal, 0/1/2/3/4; [SUM] skipped) ===")
    df_tok = pd.DataFrame(token_rows)
    alpha_tok = simpledorff.calculate_krippendorffs_alpha_for_df(
        df_tok,
        experiment_col="unit_id",
        annotator_col="annotator",
        class_col="class",
        metric_fn=simpledorff.metrics.nominal_metric,
    )
    print(f"alpha_token = {alpha_tok:.4f}")
    for t in (0, 1, 2, 3):
        df_tt = pd.DataFrame(token_rows_by_type[t])
        alpha_tt = simpledorff.calculate_krippendorffs_alpha_for_df(
            df_tt,
            experiment_col="unit_id",
            annotator_col="annotator",
            class_col="class",
            metric_fn=simpledorff.metrics.nominal_metric,
        )
        print(f"  alpha_token_type{t} (binary): {alpha_tt:.4f}")

    print("\n=== Instance-level error counts (interval; [SUM] skipped) ===")
    df_cnt = pd.DataFrame(count_rows)
    alpha_cnt = simpledorff.calculate_krippendorffs_alpha_for_df(
        df_cnt,
        experiment_col="unit_id",
        annotator_col="annotator",
        class_col="value",
        metric_fn=simpledorff.metrics.interval_metric,
    )
    print(f"alpha_count = {alpha_cnt:.4f}")
    for t in (0, 1, 2, 3):
        df_ct = pd.DataFrame(count_rows_by_type[t])
        alpha_ct = simpledorff.calculate_krippendorffs_alpha_for_df(
            df_ct,
            experiment_col="unit_id",
            annotator_col="annotator",
            class_col="value",
            metric_fn=simpledorff.metrics.interval_metric,
        )
        print(f"  alpha_count_type{t}: {alpha_ct:.4f}")

    print("\n=== System-level average error counts (interval; [SUM] skipped) ===")
    df_sys = pd.DataFrame(system_rows)
    alpha_sys = simpledorff.calculate_krippendorffs_alpha_for_df(
        df_sys,
        experiment_col="unit_id",
        annotator_col="annotator",
        class_col="value",
        metric_fn=simpledorff.metrics.interval_metric,
    )
    print(f"alpha_system = {alpha_sys:.4f}")
    for t in (0, 1, 2, 3):
        df_st = pd.DataFrame(system_rows_by_type[t])
        alpha_st = simpledorff.calculate_krippendorffs_alpha_for_df(
            df_st,
            experiment_col="unit_id",
            annotator_col="annotator",
            class_col="value",
            metric_fn=simpledorff.metrics.interval_metric,
        )
        print(f"  alpha_system_type{t}: {alpha_st:.4f}")

    print("\n=== Summarization-level agreement (nominal over short/medium/none) ===")
    df_sum = pd.DataFrame(summary_rows)
    alpha_sum = simpledorff.calculate_krippendorffs_alpha_for_df(
        df_sum,
        experiment_col="unit_id",
        annotator_col="annotator",
        class_col="value",
        metric_fn=simpledorff.metrics.nominal_metric,
    )
    print(f"alpha_summary = {alpha_sum:.4f}")


if __name__ == "__main__":
    main()


