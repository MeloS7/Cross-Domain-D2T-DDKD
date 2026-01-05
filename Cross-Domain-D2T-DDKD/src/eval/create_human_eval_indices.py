#!/usr/bin/env python3
"""
Create a human-evaluation subset from LLM-judge annotations for QUINTD datasets.

Goals:
- For each domain (OWID / gsmarena / weather / wikidata / ice_hockey), draw 12
  samples from GPT-5.1 judgements of a fixed system (Qwen3-32B variant), using
  stratified sampling:
    * 4 error types × 3 samples each:
        - type 0: Incorrect Fact
        - type 1: Not Checkable
        - type 2: Misleading
        - type 3: Other
    * Priority when selecting per type:
        (i) single-category outputs with 1–3 errors of that type;
        (ii) single-category outputs with any count of that type;
        (iii) multi-category outputs with 1–3 errors of that type;
        (iv) multi-category outputs with any count of that type;
        (v) if fewer than 12 total after the above, fill with random remaining
            samples (may include no-error samples) and report coverage.

- For each domain:
    1) Output selected sample indices (table_idx).
    2) Summarize error-type distribution for these samples under GPT-5.1 and
       Gemini-2.5-pro:
         - error span counts per type
         - number of examples containing at least one error of that type

Usage (from inner project root `Cross-Domain-D2T-DDKD/`):

    python -m Cross-Domain-D2T-DDKD.src.eval.create_human_eval_set \\
        --base_dir . \\
        --seed 42

Outputs:
- Print a JSON dict: domain -> [selected table_idx].
- Then print per-domain summaries for GPT-5.1 and Gemini-2.5-pro.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set


# Semantic labels for four error types, consistent with eval_judges_agreement.py
ERROR_TYPE_LABELS = {
    0: "Incorrect Fact",
    1: "Not Checkable",
    2: "Misleading",
    3: "Other",
}


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def analyze_examples(records: List[dict]) -> List[dict]:
    """
    Per example, compute:
      - counts[t]: number of error spans for each type t in {0,1,2,3}
      - types_present: set of error types that appear
    """
    per_example: List[dict] = []
    for idx, rec in enumerate(records):
        counts = {t: 0 for t in range(4)}
        types_present: Set[int] = set()
        for err in rec.get("annotations", []):
            t = err.get("type")
            # Ignore special markers (e.g., -1: PARSING_FAILED)
            if t in (0, 1, 2, 3):
                counts[t] += 1
                types_present.add(t)
        per_example.append(
            {
                "idx": idx,
                "counts": counts,
                "types_present": types_present,
            }
        )
    return per_example


def stratified_sample_indices(
    per_example: List[dict],
    n_per_category: int = 3,
    seed: int = 42,
) -> List[int]:
    """
    Stratified sampling to select 12 samples from one system's LLM-judge output.

    Sampling strategy (run for each error type t):
      1. single-category outputs with 1–3 errors of type t;
      2. single-category outputs with any count of type t;
      3. multi-category outputs with 1–3 errors of type t;
      4. multi-category outputs with any count of type t;
      5. If total across 4 types < 4 * n_per_category, randomly fill from
         remaining samples—prefer those with any errors, then no-error samples.

    Note: one sample can be allocated to at most one target category's quota,
    though it may contain multiple error types internally.
    """
    rng = random.Random(seed)
    cat_order = [0, 1, 2, 3]

    # Prepare four priority pools for each category
    tiers = ["single_1to3", "single_any", "multi_1to3", "multi_any"]
    pools: Dict[int, Dict[str, List[int]]]={
        c: {tier: [] for tier in tiers} for c in cat_order
    }

    all_with_error: List[int] = []
    all_no_error: List[int] = []

    for ex in per_example:
        idx = ex["idx"]
        counts = ex["counts"]
        types = ex["types_present"]
        if types:
            all_with_error.append(idx)
        else:
            all_no_error.append(idx)

        for c in cat_order:
            cnt = counts[c]
            if cnt <= 0:
                continue
            if len(types) == 1:
                # single‑category
                if 1 <= cnt <= 3:
                    pools[c]["single_1to3"].append(idx)
                else:
                    pools[c]["single_any"].append(idx)
            else:
                # multi‑category
                if 1 <= cnt <= 3:
                    pools[c]["multi_1to3"].append(idx)
                else:
                    pools[c]["multi_any"].append(idx)

    # Shuffle pools to ensure randomness
    for c in cat_order:
        for tier in tiers:
            rng.shuffle(pools[c][tier])

    selected: List[int] = []
    selected_set: Set[int] = set()

    # Allocate quota per error type
    for c in cat_order:
        needed = n_per_category
        for tier in tiers:
            if needed <= 0:
                break
            for idx in pools[c][tier]:
                if idx in selected_set:
                    continue
                selected_set.add(idx)
                selected.append(idx)
                needed -= 1
                if needed <= 0:
                    break

    target_total = n_per_category * len(cat_order)

    # If fewer than 12 samples selected, fill randomly from remaining
    if len(selected) < target_total:
        remaining = target_total - len(selected)

        # Fill from samples that contain any errors first
        remaining_error = [i for i in all_with_error if i not in selected_set]
        rng.shuffle(remaining_error)
        for idx in remaining_error:
            if remaining <= 0:
                break
            selected_set.add(idx)
            selected.append(idx)
            remaining -= 1

        # If still short, fill from no-error samples
        if remaining > 0:
            remaining_clean = [i for i in all_no_error if i not in selected_set]
            rng.shuffle(remaining_clean)
            for idx in remaining_clean:
                if remaining <= 0:
                    break
                selected_set.add(idx)
                selected.append(idx)
                remaining -= 1

    selected.sort()
    return selected


def stratified_sample_indices_two_anchors(
    per_example_primary: List[dict],
    per_example_fallback: List[dict],
    n_per_category: int = 3,
    seed: int = 42,
) -> List[int]:
    """
    Stratified sampling with two anchors:
      - primary: best model (preferred)
      - fallback: weakest zero-shot model (used only if primary lacks a type)
    Other logic matches stratified_sample_indices.
    """
    rng = random.Random(seed)
    cat_order = [0, 1, 2, 3]
    tiers = ["single_1to3", "single_any", "multi_1to3", "multi_any"]

    def build_pools(per_example: List[dict]):
        pools: Dict[int, Dict[str, List[int]]] = {
            c: {tier: [] for tier in tiers} for c in cat_order
        }
        all_with_error: List[int] = []
        all_no_error: List[int] = []

        for ex in per_example:
            idx = ex["idx"]
            counts = ex["counts"]
            types = ex["types_present"]
            if types:
                all_with_error.append(idx)
            else:
                all_no_error.append(idx)

            for c in cat_order:
                cnt = counts[c]
                if cnt <= 0:
                    continue
                if len(types) == 1:
                    if 1 <= cnt <= 3:
                        pools[c]["single_1to3"].append(idx)
                    else:
                        pools[c]["single_any"].append(idx)
                else:
                    if 1 <= cnt <= 3:
                        pools[c]["multi_1to3"].append(idx)
                    else:
                        pools[c]["multi_any"].append(idx)

        for c in cat_order:
            for tier in tiers:
                rng.shuffle(pools[c][tier])

        return pools, all_with_error, all_no_error

    pools_primary, all_err_primary, all_clean_primary = build_pools(per_example_primary)
    pools_fallback, all_err_fallback, all_clean_fallback = build_pools(per_example_fallback)

    selected: List[int] = []
    selected_set: Set[int] = set()

        # Draw per category from primary, then supplement with fallback if needed
    for c in cat_order:
        needed = n_per_category
        # primary
        for tier in tiers:
            if needed <= 0:
                break
            for idx in pools_primary[c][tier]:
                if idx in selected_set:
                    continue
                selected_set.add(idx)
                selected.append(idx)
                needed -= 1
                if needed <= 0:
                    break
        # fallback
        if needed > 0:
            for tier in tiers:
                if needed <= 0:
                    break
                for idx in pools_fallback[c][tier]:
                    if idx in selected_set:
                        continue
                    selected_set.add(idx)
                    selected.append(idx)
                    needed -= 1
                    if needed <= 0:
                        break

    target_total = n_per_category * len(cat_order)

    # If still below target, fill using remaining samples from primary+fallback
    if len(selected) < target_total:
        remaining = target_total - len(selected)
        all_err_union = set(all_err_primary) | set(all_err_fallback)
        remaining_error = [i for i in all_err_union if i not in selected_set]
        rng.shuffle(remaining_error)
        for idx in remaining_error:
            if remaining <= 0:
                break
            selected_set.add(idx)
            selected.append(idx)
            remaining -= 1

        if remaining > 0:
            all_clean_union = set(all_clean_primary) | set(all_clean_fallback)
            remaining_clean = [i for i in all_clean_union if i not in selected_set]
            rng.shuffle(remaining_clean)
            for idx in remaining_clean:
                if remaining <= 0:
                    break
                selected_set.add(idx)
                selected.append(idx)
                remaining -= 1

    selected.sort()
    return selected


def compute_error_stats_for_indices(
    records: List[dict],
    indices: List[int],
) -> Dict[str, Dict[int, int]]:
    """
    For a set of indices under one judge, summarize error distributions.

    Returns:
      {
        "error_spans": {t: count of error spans},
        "examples_with_type": {t: number of examples containing at least one error of type t},
      }
    """
    error_spans = defaultdict(int)          # t -> count of error spans
    examples_with_type = defaultdict(int)   # t -> #examples containing at least one error of type t

    for idx in indices:
        if idx < 0 or idx >= len(records):
            continue
        rec = records[idx]
        ann = rec.get("annotations", [])
        seen_types: Set[int] = set()
        for err in ann:
            t = err.get("type")
            if t in (0, 1, 2, 3):
                error_spans[t] += 1
                seen_types.add(t)
        for t in seen_types:
            examples_with_type[t] += 1

    # Ensure keys 0–3 exist to avoid missing entries
    for t in range(4):
        _ = error_spans[t]
        _ = examples_with_type[t]

    return {
        "error_spans": dict(error_spans),
        "examples_with_type": dict(examples_with_type),
    }


def compute_error_stats_for_indices_multi_system(
    systems_records: Dict[str, List[dict]],
    indices: List[int],
) -> Dict[str, Dict[int, int]]:
    """
    For a set of indices, aggregate error distributions across multiple systems
    under GPT-5.1 judgments.

    - error_spans[t]: total error spans of type t across systems and selected examples
    - examples_with_type[t]: count of system×example pairs containing at least one error of type t
    """
    error_spans = defaultdict(int)
    examples_with_type = defaultdict(int)

    for _sys_name, records in systems_records.items():
        for idx in indices:
            if idx < 0 or idx >= len(records):
                continue
            rec = records[idx]
            ann = rec.get("annotations", [])
            seen_types: Set[int] = set()
            for err in ann:
                t = err.get("type")
                if t in (0, 1, 2, 3):
                    error_spans[t] += 1
                    seen_types.add(t)
            for t in seen_types:
                examples_with_type[t] += 1

    for t in range(4):
        _ = error_spans[t]
        _ = examples_with_type[t]

    return {
        "error_spans": dict(error_spans),
        "examples_with_type": dict(examples_with_type),
    }


def pretty_print_domain_stats(
    domain: str,
    judge_name: str,
    stats: Dict[str, Dict[int, int]],
    total_examples: int,
) -> None:
    print(f"\n--- Domain: {domain} | Judge: {judge_name} ---")
    print(f"Total selected examples: {total_examples}")
    print(f"{'Type':25s} {'Span#':>8s} {'Example#':>10s}")
    print("-" * 48)
    spans = stats["error_spans"]
    exs = stats["examples_with_type"]
    for t in range(4):
        label = ERROR_TYPE_LABELS.get(t, f"type{t}")
        s = spans.get(t, 0)
        e = exs.get(t, 0)
        print(f"{f'type{t} ({label})':25s} {s:8d} {e:10d}")


def build_default_domain_config(base_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Build default file-path configuration for the five domains.

    Returns:
      {
        domain: {
          "gpt":    <gpt-5.1 jsonl path>,
          "gemini": <gemini-2.5-pro jsonl path>,
        },
        ...
      }
    """
    # Paths are relative to inner project root `Cross-Domain-D2T-DDKD/`
    rel_paths_gpt = {
        "owid": "data/model_outputs/quintd/owid/eval_res/sft_lora/owid_sft_lora_on_webnlg_csv_qwen3_32b_gpt-5.1.jsonl",
        "gsmarena": "data/model_outputs/quintd/gsmarena/eval_res/zero_shot/gsmarena_zero_shot_qwen3_32b_gpt-5.1.jsonl",
        "weather": "data/model_outputs/quintd/weather/eval_res/sft_lora/weather_sft_lora_on_webnlg_json_qwen3_32B_gpt-5.1.jsonl",
        "wikidata": "data/model_outputs/quintd/wikidata/eval_res/sft_lora/wikidata_sft_lora_on_webnlg_mkd_qwen3_32B_gpt-5.1.jsonl",
        "ice_hockey": "data/model_outputs/quintd/ice_hockey/eval_res/sft_lora/ice_hockey_sft_lora_on_webnlg_json_qwen3_32B_gpt-5.1.jsonl",
    }

    config: Dict[str, Dict[str, str]] = {}
    for domain, rel_gpt in rel_paths_gpt.items():
        gpt_path = os.path.join(base_dir, rel_gpt)
        gemini_path = gpt_path.replace("gpt-5.1", "gemini-2.5-pro")
        config[domain] = {"gpt": gpt_path, "gemini": gemini_path}
    return config


def build_additional_domain_paths(base_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Provide extra GPT-5.1 evaluation paths per domain:
      - fallback_gpt: weakest model (Qwen3-1.7B zero_shot)
      - sft_small_gpt: small SFT-LoRA model (Qwen3-1.7B)
      - ddkd_best_gpt: best distilled model (DDKD)
    """
    def p(rel: str) -> str:
        return os.path.join(base_dir, rel)

    return {
        "owid": {
            "fallback_gpt": p(
                "data/model_outputs/quintd/owid/eval_res/zero_shot/owid_zero_shot_qwen3_1.7b_gpt-5.1.jsonl"
            ),
            "sft_small_gpt": p(
                "data/model_outputs/quintd/owid/eval_res/sft_lora/owid_sft_lora_on_webnlg_csv_qwen3_1.7b_gpt-5.1.jsonl"
            ),
            "ddkd_best_gpt": p(
                "data/model_outputs/quintd/owid/eval_res/ddkd_sft_lora/owid_ddkd_sft_lora_qwen3_1.7b_mixed_gpt-5.1.jsonl"
            ),
        },
        "gsmarena": {
            "fallback_gpt": p(
                "data/model_outputs/quintd/gsmarena/eval_res/zero_shot/gsmarena_zero_shot_qwen3_1.7b_gpt-5.1.jsonl"
            ),
            "sft_small_gpt": p(
                "data/model_outputs/quintd/gsmarena/eval_res/sft_lora/gsmarena_sft_lora_on_webnlg_json_qwen3_1.7b_gpt-5.1.jsonl"
            ),
            "ddkd_best_gpt": p(
                "data/model_outputs/quintd/gsmarena/eval_res/ddkd_zero_shot/gsmarena_ddkd_zero_shot_qwen3_1.7b_mixed_gpt-5.1.jsonl"
            ),
        },
        "weather": {
            "fallback_gpt": p(
                "data/model_outputs/quintd/weather/eval_res/zero_shot/weather_zero_shot_qwen3_1.7b_gpt-5.1.jsonl"
            ),
            "sft_small_gpt": p(
                "data/model_outputs/quintd/weather/eval_res/sft_lora/weather_sft_lora_on_webnlg_json_qwen3_1.7b_gpt-5.1.jsonl"
            ),
            "ddkd_best_gpt": p(
                "data/model_outputs/quintd/weather/eval_res/ddkd_sft_lora/weather_ddkd_sft_lora_qwen3_1.7b_sub_gpt-5.1.jsonl"
            ),
        },
        "wikidata": {
            "fallback_gpt": p(
                "data/model_outputs/quintd/wikidata/eval_res/zero_shot/wikidata_zero_shot_qwen3_1.7b_gpt-5.1.jsonl"
            ),
            "sft_small_gpt": p(
                "data/model_outputs/quintd/wikidata/eval_res/sft_lora/wikidata_sft_lora_on_webnlg_mkd_qwen3_1.7b_gpt-5.1.jsonl"
            ),
            "ddkd_best_gpt": p(
                "data/model_outputs/quintd/wikidata/eval_res/ddkd_sft_lora/wikidata_ddkd_sft_lora_qwen3_32B_mixed_gpt-5.1.jsonl"
            ),
        },
        "ice_hockey": {
            "fallback_gpt": p(
                "data/model_outputs/quintd/ice_hockey/eval_res/zero_shot/ice_hockey_zero_shot_qwen3_1.7b_gpt-5.1.jsonl"
            ),
            "sft_small_gpt": p(
                "data/model_outputs/quintd/ice_hockey/eval_res/sft_lora/ice_hockey_sft_lora_on_webnlg_json_qwen3_1.7b_gpt-5.1.jsonl"
            ),
            "ddkd_best_gpt": p(
                "data/model_outputs/quintd/ice_hockey/eval_res/ddkd_sft_lora/ice_hockey_ddkd_sft_lora_qwen3_1.7b_mixed_gpt-5.1.jsonl"
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a human evaluation subset (12 examples per domain) "
            "from GPT‑5.1 LLM‑judge outputs, and compare error distributions "
            "with Gemini‑2.5‑pro."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Base directory where 'data/' lives (default: current directory).",
    )
    parser.add_argument(
        "--output_indices",
        type=str,
        default="data/test/quintd/human_eval_indices_qwen3_32B_gpt-5.1.json",
        help=(
            "Path to save selected indices as a JSON dict "
            "'{domain: [table_idx, ...]}'. "
            "If relative, it is interpreted w.r.t. base_dir."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    seed = args.seed

    domain_config = build_default_domain_config(base_dir)
    extra_domain_paths = build_additional_domain_paths(base_dir)
    output_indices_path = args.output_indices

    selected_indices_per_domain: Dict[str, List[int]] = {}

    for domain, paths in domain_config.items():
        gpt_path = paths["gpt"]
        gemini_path = paths["gemini"]

        extra = extra_domain_paths.get(domain)
        if extra is None:
            print(f"[Warning] No extra paths configured for domain '{domain}', skip.")
            continue

        fallback_gpt_path = extra["fallback_gpt"]
        sft_small_gpt_path = extra["sft_small_gpt"]
        ddkd_best_gpt_path = extra["ddkd_best_gpt"]

        if not os.path.isfile(gpt_path):
            print(f"[Warning] GPT‑5.1 file not found for domain '{domain}': {gpt_path}")
            continue
        if not os.path.isfile(gemini_path):
            print(
                f"[Warning] Gemini‑2.5‑pro file not found for domain '{domain}': "
                f"{gemini_path}"
            )
            continue
        if not os.path.isfile(fallback_gpt_path):
            print(
                f"[Warning] Fallback GPT‑5.1 file not found for domain '{domain}': "
                f"{fallback_gpt_path}"
            )
            continue
        if not os.path.isfile(sft_small_gpt_path):
            print(
                f"[Warning] Small SFT GPT‑5.1 file not found for domain '{domain}': "
                f"{sft_small_gpt_path}"
            )
            continue
        if not os.path.isfile(ddkd_best_gpt_path):
            print(
                f"[Warning] DDKD GPT‑5.1 file not found for domain '{domain}': "
                f"{ddkd_best_gpt_path}"
            )
            continue

        # primary system (best model)
        gpt_records = load_jsonl(gpt_path)
        gemini_records = load_jsonl(gemini_path)

        # fallback + other systems (all GPT‑5.1 judges)
        fallback_gpt_records = load_jsonl(fallback_gpt_path)
        sft_small_gpt_records = load_jsonl(sft_small_gpt_path)
        ddkd_best_gpt_records = load_jsonl(ddkd_best_gpt_path)

        # Corresponding Gemini-2.5-pro evaluation paths (via suffix replacement)
        fallback_gemini_path = fallback_gpt_path.replace("gpt-5.1", "gemini-2.5-pro")
        sft_small_gemini_path = sft_small_gpt_path.replace("gpt-5.1", "gemini-2.5-pro")
        ddkd_best_gemini_path = ddkd_best_gpt_path.replace("gpt-5.1", "gemini-2.5-pro")

        if not os.path.isfile(fallback_gemini_path):
            print(
                f"[Warning] Fallback Gemini‑2.5‑pro file not found for domain '{domain}': "
                f"{fallback_gemini_path}"
            )
            continue
        if not os.path.isfile(sft_small_gemini_path):
            print(
                f"[Warning] Small SFT Gemini‑2.5‑pro file not found for domain '{domain}': "
                f"{sft_small_gemini_path}"
            )
            continue
        if not os.path.isfile(ddkd_best_gemini_path):
            print(
                f"[Warning] DDKD Gemini‑2.5‑pro file not found for domain '{domain}': "
                f"{ddkd_best_gemini_path}"
            )
            continue

        fallback_gemini_records = load_jsonl(fallback_gemini_path)
        sft_small_gemini_records = load_jsonl(sft_small_gemini_path)
        ddkd_best_gemini_records = load_jsonl(ddkd_best_gemini_path)

        if len(gpt_records) != len(gemini_records):
            print(
                f"[Warning] Length mismatch for domain '{domain}': "
                f"gpt-5.1={len(gpt_records)}, gemini-2.5-pro={len(gemini_records)}"
            )

        per_example_primary = analyze_examples(gpt_records)
        per_example_fallback = analyze_examples(fallback_gpt_records)
        indices = stratified_sample_indices_two_anchors(
            per_example_primary=per_example_primary,
            per_example_fallback=per_example_fallback,
            n_per_category=3,
            seed=seed,
        )
        selected_indices_per_domain[domain] = indices

        # Aggregate error distributions for the same samples across four systems
        systems_gpt_records = {
            "primary": gpt_records,
            "zero_shot_1.7B": fallback_gpt_records,
            "sft_lora_1.7B": sft_small_gpt_records,
            "ddkd_best": ddkd_best_gpt_records,
        }
        systems_gemini_records = {
            "primary": gemini_records,
            "zero_shot_1.7B": fallback_gemini_records,
            "sft_lora_1.7B": sft_small_gemini_records,
            "ddkd_best": ddkd_best_gemini_records,
        }

        multi_stats_gpt = compute_error_stats_for_indices_multi_system(
            systems_gpt_records, indices
        )
        pretty_print_domain_stats(
            domain=domain,
            judge_name="GPT-5.1 (4 systems aggregated)",
            stats=multi_stats_gpt,
            total_examples=len(indices) * len(systems_gpt_records),
        )

        multi_stats_gemini = compute_error_stats_for_indices_multi_system(
            systems_gemini_records, indices
        )
        pretty_print_domain_stats(
            domain=domain,
            judge_name="Gemini-2.5-pro (4 systems aggregated)",
            stats=multi_stats_gemini,
            total_examples=len(indices) * len(systems_gpt_records),
        )

    # Finally write JSON dict: domain -> [selected table_idx] for downstream use
    out_path = output_indices_path
    if not os.path.isabs(out_path):
        out_path = os.path.join(base_dir, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            selected_indices_per_domain,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    print(f"\nSaved selected indices to JSON file: {out_path}")


if __name__ == "__main__":
    main()


