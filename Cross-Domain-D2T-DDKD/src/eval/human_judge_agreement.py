#!/usr/bin/env python3
"""
Compute agreement between aggregated human annotations (3 annotators) and GPT-5.1
at instance-level and system-level on the 240-instance human eval set.

Data:
  - Human annotations:
      data/human_eval/human_annotations_annotator_{z,c,y}.jsonl
  - Selected indices (per domain):
      data/human_eval/human_eval_indices.json
  - GPT-5.1 eval files (per domain, 4 systems):
      paths follow the same conventions as create_human_eval_indices.py

Metrics:
  - Instance-level: Pearson r between human-mean error counts and GPT counts
    for each type (0/1/2/3) and all.
  - System-level: For each type/all, build vectors of per-system average error
    counts (over the 12 selected instances) for humans vs GPT, then Pearson r
    across the 4 systems.

Notes:
  - [SUM] entries are ignored for error counting.
  - Type=0 is a valid error type; all types {0,1,2,3} are counted separately,
    and "all" is the sum of the four types.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ANNOTATORS = ["z", "c", "y"]
TYPES = [0, 1, 2, 3]


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pearson_corr(x: List[float], y: List[float]) -> Tuple[float, str]:
    if len(x) == 0 or len(x) != len(y):
        return float("nan"), "empty or length mismatch"
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = 0.0
    denom_x = 0.0
    denom_y = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mean_x
        dy = yi - mean_y
        num += dx * dy
        denom_x += dx * dx
        denom_y += dy * dy
    if denom_x == 0 and denom_y == 0:
        return float("nan"), "both constant (zero variance)"
    if denom_x == 0:
        return float("nan"), "human constant (zero variance)"
    if denom_y == 0:
        return float("nan"), "gpt constant (zero variance)"
    return num / math.sqrt(denom_x * denom_y), ""


def human_counts_per_instance(human_data: Dict[str, dict]) -> Dict[str, Dict[str, float]]:
    """
    Return instance_id -> { 'type0':float, 'type1':..., 'type2':..., 'type3':..., 'all': float }
    by averaging counts across annotators (only over available annotators per instance).
    """
    # group by instance_id
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for inst_id, rec in human_data.items():
        grouped[inst_id].append(rec)

    result: Dict[str, Dict[str, float]] = {}
    for inst_id, recs in grouped.items():
        counts = {t: [] for t in TYPES}
        for rec in recs:
            errs = [
                e
                for e in rec.get("errors", [])
                if not (
                    isinstance(e.get("text", ""), str)
                    and e.get("text", "").strip().lower() == "[sum]"
                )
            ]
            for t in TYPES:
                cnt_t = sum(1 for e in errs if e.get("type") == t)
                counts[t].append(cnt_t)
        avg_counts = {t: (sum(v) / len(v) if v else 0.0) for t, v in counts.items()}
        avg_counts["all"] = sum(avg_counts[t] for t in TYPES)
        result[inst_id] = avg_counts
    return result


def gpt_counts_per_instance(records: List[dict]) -> Dict[int, Dict[str, float]]:
    """
    Return table_idx -> counts dict per type and all.
    """
    out: Dict[int, Dict[str, float]] = {}
    for rec in records:
        idx = rec.get("table_idx")
        if idx is None:
            continue
        errs = rec.get("annotations", []) or rec.get("annotations", []) or rec.get("annotations", [])
        counts = {t: 0 for t in TYPES}
        for err in errs:
            t = err.get("type")
            if t in TYPES:
                counts[t] += 1
        counts["all"] = sum(counts[t] for t in TYPES)
        out[int(idx)] = counts
    return out


def build_paths(base_dir: Path) -> Dict[str, Dict[str, str]]:
    """
    Build GPT-5.1 eval paths for four systems per domain:
      primary, zero_shot_1.7B, sft_lora_1.7B, ddkd_best
    """
    def p(rel: str) -> str:
        return str(base_dir / rel)

    cfg = {
        "owid": {
            "primary": p("data/model_outputs/quintd/owid/eval_res/sft_lora/owid_sft_lora_on_webnlg_csv_qwen3_32b_gpt-5.1.jsonl"),
            "zero_shot_1.7B": p("data/model_outputs/quintd/owid/eval_res/zero_shot/owid_zero_shot_qwen3_1.7b_gpt-5.1.jsonl"),
            "sft_lora_1.7B": p("data/model_outputs/quintd/owid/eval_res/sft_lora/owid_sft_lora_on_webnlg_csv_qwen3_1.7b_gpt-5.1.jsonl"),
            "ddkd_best": p("data/model_outputs/quintd/owid/eval_res/ddkd_sft_lora/owid_ddkd_sft_lora_qwen3_1.7b_mixed_gpt-5.1.jsonl"),
        },
        "gsmarena": {
            "primary": p("data/model_outputs/quintd/gsmarena/eval_res/zero_shot/gsmarena_zero_shot_qwen3_32b_gpt-5.1.jsonl"),
            "zero_shot_1.7B": p("data/model_outputs/quintd/gsmarena/eval_res/zero_shot/gsmarena_zero_shot_qwen3_1.7b_gpt-5.1.jsonl"),
            "sft_lora_1.7B": p("data/model_outputs/quintd/gsmarena/eval_res/sft_lora/gsmarena_sft_lora_on_webnlg_json_qwen3_1.7b_gpt-5.1.jsonl"),
            "ddkd_best": p("data/model_outputs/quintd/gsmarena/eval_res/ddkd_zero_shot/gsmarena_ddkd_zero_shot_qwen3_1.7b_mixed_gpt-5.1.jsonl"),
        },
        "weather": {
            "primary": p("data/model_outputs/quintd/weather/eval_res/sft_lora/weather_sft_lora_on_webnlg_json_qwen3_32B_gpt-5.1.jsonl"),
            "zero_shot_1.7B": p("data/model_outputs/quintd/weather/eval_res/zero_shot/weather_zero_shot_qwen3_1.7B_gpt-5.1.jsonl"),
            "sft_lora_1.7B": p("data/model_outputs/quintd/weather/eval_res/sft_lora/weather_sft_lora_on_webnlg_json_qwen3_1.7B_gpt-5.1.jsonl"),
            "ddkd_best": p("data/model_outputs/quintd/weather/eval_res/ddkd_sft_lora/weather_ddkd_sft_lora_qwen3_1.7b_sub_gpt-5.1.jsonl"),
        },
        "wikidata": {
            "primary": p("data/model_outputs/quintd/wikidata/eval_res/sft_lora/wikidata_sft_lora_on_webnlg_mkd_qwen3_32B_gpt-5.1.jsonl"),
            "zero_shot_1.7B": p("data/model_outputs/quintd/wikidata/eval_res/zero_shot/wikidata_zero_shot_qwen3_1.7B_gpt-5.1.jsonl"),
            "sft_lora_1.7B": p("data/model_outputs/quintd/wikidata/eval_res/sft_lora/wikidata_sft_lora_on_webnlg_mkd_qwen3_1.7B_gpt-5.1.jsonl"),
            "ddkd_best": p("data/model_outputs/quintd/wikidata/eval_res/ddkd_sft_lora/wikidata_ddkd_sft_lora_qwen3_32B_mixed_gpt-5.1.jsonl"),
        },
        "ice_hockey": {
            "primary": p("data/model_outputs/quintd/ice_hockey/eval_res/sft_lora/ice_hockey_sft_lora_on_webnlg_json_qwen3_32B_gpt-5.1.jsonl"),
            "zero_shot_1.7B": p("data/model_outputs/quintd/ice_hockey/eval_res/zero_shot/ice_hockey_zero_shot_qwen3_1.7b_gpt-5.1.jsonl"),
            "sft_lora_1.7B": p("data/model_outputs/quintd/ice_hockey/eval_res/sft_lora/ice_hockey_sft_lora_on_webnlg_json_qwen3_1.7b_gpt-5.1.jsonl"),
            "ddkd_best": p("data/model_outputs/quintd/ice_hockey/eval_res/ddkd_sft_lora/ice_hockey_ddkd_sft_lora_qwen3_1.7b_mixed_gpt-5.1.jsonl"),
        },
    }
    return cfg


# Map human-prettified system names (from streamlit_annotation pretty_system_name)
# to GPT system keys, per domain.
HUMAN_SYS_MAP = {
    "gsmarena": {
        "zero_shot_32B": "primary",
        "zero_shot_1.7B": "zero_shot_1.7B",
        "sft_webnlg_1.7B": "sft_lora_1.7B",
        "distilled_from_zero_shot_32B_mixed": "ddkd_best",
        # also allow raw keys
        "primary_32B": "primary",
        "sft_lora_1.7B": "sft_lora_1.7B",
        "ddkd_best": "ddkd_best",
    },
    "weather": {
        "sft_webnlg_32B": "primary",
        "zero_shot_1.7B": "zero_shot_1.7B",
        "sft_webnlg_1.7B": "sft_lora_1.7B",
        "distilled_from_sft_webnlg_32B_sub": "ddkd_best",
        "primary_32B": "primary",
        "sft_lora_1.7B": "sft_lora_1.7B",
        "ddkd_best": "ddkd_best",
    },
    "owid": {
        "sft_webnlg_32B": "primary",
        "zero_shot_1.7B": "zero_shot_1.7B",
        "sft_webnlg_1.7B": "sft_lora_1.7B",
        "distilled_from_sft_webnlg_32B_mixed": "ddkd_best",
        "primary_32B": "primary",
        "sft_lora_1.7B": "sft_lora_1.7B",
        "ddkd_best": "ddkd_best",
    },
    "wikidata": {
        "sft_webnlg_32B": "primary",
        "zero_shot_1.7B": "zero_shot_1.7B",
        "sft_webnlg_1.7B": "sft_lora_1.7B",
        "distilled_from_sft_webnlg_32B_mixed": "ddkd_best",
        "primary_32B": "primary",
        "sft_lora_1.7B": "sft_lora_1.7B",
        "ddkd_best": "ddkd_best",
    },
    "ice_hockey": {
        "sft_webnlg_32B": "primary",
        "zero_shot_1.7B": "zero_shot_1.7B",
        "sft_webnlg_1.7B": "sft_lora_1.7B",
        "distilled_from_sft_webnlg_32B_mixed": "ddkd_best",
        "primary_32B": "primary",
        "sft_lora_1.7B": "sft_lora_1.7B",
        "ddkd_best": "ddkd_best",
    },
}

# Debug helper: print paired vectors for a given domain and type (default type0)
def debug_print_vectors(
    domain: str,
    idx_list: List[int],
    gpt_by_sys: Dict[str, Dict[int, Dict[str, float]]],
    human_mean_counts: Dict[str, Dict[str, float]],
    type_key: int = 0,
):
    print(f"\n[Debug] Domain={domain}, type={type_key} instance-level paired values")
    for sys_name, gpt_counts in gpt_by_sys.items():
        hv = []
        gv = []
        for inst_id, h_counts in human_mean_counts.items():
            parts = inst_id.split("|")
            if len(parts) != 3:
                continue
            dom_h, table_idx_str, human_sys = parts
            if dom_h != domain:
                continue
            mapped_sys = HUMAN_SYS_MAP.get(domain, {}).get(human_sys)
            if mapped_sys != sys_name:
                continue
            try:
                table_idx = int(table_idx_str)
            except ValueError:
                continue
            if table_idx not in idx_list:
                continue
            g_counts = gpt_counts.get(table_idx)
            if g_counts is None:
                continue
            hv.append(h_counts.get(type_key, 0.0))
            gv.append(g_counts.get(type_key, 0.0))
        print(f"  System {sys_name}: matched {len(hv)} samples")
        print(f"    human: {hv}")
        print(f"    gpt  : {gv}")

    # System-level vectors (averages)
    print(f"[Debug] Domain={domain}, type={type_key} system-level paired averages")
    hv_sys = []
    gv_sys = []
    for sys_name, gpt_counts in gpt_by_sys.items():
        agg_h = 0.0
        agg_g = 0.0
        n_used = 0
        for inst_id, h_counts in human_mean_counts.items():
            parts = inst_id.split("|")
            if len(parts) != 3:
                continue
            dom_h, table_idx_str, human_sys = parts
            if dom_h != domain:
                continue
            mapped_sys = HUMAN_SYS_MAP.get(domain, {}).get(human_sys)
            if mapped_sys != sys_name:
                continue
            try:
                table_idx = int(table_idx_str)
            except ValueError:
                continue
            if table_idx not in idx_list:
                continue
            g_counts = gpt_counts.get(table_idx)
            if g_counts is None:
                continue
            agg_h += h_counts.get(type_key, 0.0)
            agg_g += g_counts.get(type_key, 0.0)
            n_used += 1
        if n_used == 0:
            print(f"  System {sys_name}: no matched samples for system-level")
            continue
        hv_sys.append(agg_h / n_used)
        gv_sys.append(agg_g / n_used)
        print(f"  System {sys_name}: n={n_used}, human_avg={agg_h/n_used:.4f}, gpt_avg={agg_g/n_used:.4f}")
    print(f"  human system vector: {hv_sys}")
    print(f"  gpt   system vector: {gv_sys}")


def main():
    parser = argparse.ArgumentParser(
        description="Agreement between aggregated human annotations and GPT-5.1 (instance/system level)."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Base dir where 'data/' lives (default: current directory).",
    )
    args = parser.parse_args()

    base = Path(args.base_dir)

    # Load human annotations
    human_data: Dict[str, dict] = {}
    for ann in ANNOTATORS:
        path = base / f"data/human_eval/human_annotations_annotator_{ann}.jsonl"
        if not path.is_file():
            print(f"[Warning] missing human file: {path}")
            continue
        recs = load_jsonl(path)
        for r in recs:
            inst_id = r.get("instance_id")
            if inst_id:
                human_data.setdefault(inst_id, {})
                human_data[inst_id][ann] = r

    # Average human counts per instance
    # Merge per annotator records -> average counts
    merged_human: Dict[str, dict] = {}
    for inst_id, recs_by_ann in human_data.items():
        merged = {"instance_id": inst_id, "records": list(recs_by_ann.values())}
        merged_human[inst_id] = merged

    # Build averaged counts
    human_mean_counts: Dict[str, Dict[str, float]] = {}
    for inst_id, obj in merged_human.items():
        recs = obj["records"]
        counts = {t: [] for t in TYPES}
        for rec in recs:
            errs = [
                e for e in rec.get("errors", [])
                if not (isinstance(e.get("text", ""), str) and e.get("text", "").strip().lower() == "[sum]")
            ]
            for t in TYPES:
                cnt_t = sum(1 for e in errs if e.get("type") == t)
                counts[t].append(cnt_t)
        avg = {t: (sum(v) / len(v) if v else 0.0) for t, v in counts.items()}
        avg["all"] = sum(avg[t] for t in TYPES)
        human_mean_counts[inst_id] = avg

    # Load indices
    indices_path = base / "data/human_eval/human_eval_indices.json"
    if not indices_path.is_file():
        raise SystemExit(f"indices file not found: {indices_path}")
    with indices_path.open("r", encoding="utf-8") as f:
        indices_dict = json.load(f)

    # GPT paths
    gpt_paths = build_paths(base)

    # Collect human average error counts (all types) per system × domain
    human_domain_avgs: Dict[str, Dict[str, float]] = {}

    for domain, idx_list in indices_dict.items():
        if domain not in gpt_paths:
            print(f"[Warning] no GPT paths for domain {domain}, skip.")
            continue
        systems = gpt_paths[domain]

        # Load GPT records per system
        gpt_by_sys: Dict[str, Dict[int, Dict[str, float]]] = {}
        for sys_name, path in systems.items():
            p = Path(path)
            if not p.is_file():
                print(f"[Warning] missing GPT file for domain={domain}, system={sys_name}: {p}")
                continue
            recs = load_jsonl(p)
            gpt_by_sys[sys_name] = gpt_counts_per_instance(recs)

        if len(gpt_by_sys) < 4:
            print(f"[Warning] domain {domain}: fewer than 4 systems loaded, continuing with available ones.")

        # Instance-level agreement per system
        print(f"\n===== Domain: {domain} =====")
        for sys_name, gpt_counts in gpt_by_sys.items():
            # match human system ids for this domain/system
            # build list of (human_inst_id, table_idx) that map to this sys
            inst_h: Dict[str, List[float]] = {k: [] for k in ["all"] + TYPES}
            inst_g: Dict[str, List[float]] = {k: [] for k in ["all"] + TYPES}
            n_matched = 0

            # human instance_ids are domain|table_idx|human_sys_name
            # we map human_sys_name to (domain, gpt_sys_key)
            # only keep those that map to current (domain, sys_name)
            for inst_id, h_counts in human_mean_counts.items():
                parts = inst_id.split("|")
                if len(parts) != 3:
                    continue
                dom_h, table_idx_str, human_sys = parts
                if dom_h != domain:
                    continue
                mapped_sys = HUMAN_SYS_MAP.get(domain, {}).get(human_sys)
                if mapped_sys is None:
                    continue
                if mapped_sys != sys_name:
                    continue
                try:
                    table_idx = int(table_idx_str)
                except ValueError:
                    continue
                g_counts = gpt_counts.get(table_idx)
                if g_counts is None:
                    continue
                n_matched += 1
                for t in TYPES:
                    inst_h[t].append(h_counts.get(t, 0.0))
                    inst_g[t].append(g_counts.get(t, 0.0))
                inst_h["all"].append(h_counts.get("all", 0.0))
                inst_g["all"].append(g_counts.get("all", 0.0))

            # vectors per type/all
            print(f"\n-- System: {sys_name} (instance-level Pearson r) --")
            if n_matched == 0:
                print("  [Info] No matched instances between human and GPT for this system. r=nan.")
            for t in ["all"] + TYPES:
                r, msg = pearson_corr(inst_h[t], inst_g[t])
                if math.isnan(r) and msg:
                    print(f"  type {t}: r = nan ({msg}) (n={len(inst_h[t])})")
                else:
                    print(f"  type {t}: r = {r:.4f} (n={len(inst_h[t])})")

        # System-level agreement (across systems) per type/all
        # For each system, average over its matched instances (human_mean vs gpt)
        sys_vec_h: Dict[str, List[float]] = {k: [] for k in ["all"] + TYPES}
        sys_vec_g: Dict[str, List[float]] = {k: [] for k in ["all"] + TYPES}
        for sys_name, gpt_counts in gpt_by_sys.items():
            agg_h = {k: 0.0 for k in ["all"] + TYPES}
            agg_g = {k: 0.0 for k in ["all"] + TYPES}
            n_used = 0
            # collect matched instances for this system
            matched = []
            for inst_id, h_counts in human_mean_counts.items():
                parts = inst_id.split("|")
                if len(parts) != 3:
                    continue
                dom_h, table_idx_str, human_sys = parts
                if dom_h != domain:
                    continue
                mapped_sys = HUMAN_SYS_MAP.get(domain, {}).get(human_sys)
                if mapped_sys is None:
                    continue
                if mapped_sys != sys_name:
                    continue
                try:
                    table_idx = int(table_idx_str)
                except ValueError:
                    continue
                g_counts = gpt_counts.get(table_idx)
                if g_counts is None:
                    continue
                matched.append((h_counts, g_counts))
            for h_counts, g_counts in matched:
                for t in TYPES:
                    agg_h[t] += h_counts.get(t, 0.0)
                    agg_g[t] += g_counts.get(t, 0.0)
                agg_h["all"] += h_counts.get("all", 0.0)
                agg_g["all"] += g_counts.get("all", 0.0)
                n_used += 1
            if n_used == 0:
                print(f"  [Info] No matched instances for system-level aggregation: {sys_name}")
                continue
            for k in ["all"] + TYPES:
                agg_h[k] /= n_used
                agg_g[k] /= n_used
                sys_vec_h[k].append(agg_h[k])
                sys_vec_g[k].append(agg_g[k])

        print("\n-- System-level Pearson r across systems --")
        for k in ["all"] + TYPES:
            r, msg = pearson_corr(sys_vec_h[k], sys_vec_g[k])
            if math.isnan(r) and msg:
                print(f"  type {k}: r = nan ({msg}) (n_systems={len(sys_vec_h[k])})")
            else:
                print(f"  type {k}: r = {r:.4f} (n_systems={len(sys_vec_h[k])})")

        # Optional debug print for wikidata type0
        # if domain == "wikidata":
        #     debug_print_vectors(domain, idx_list, gpt_by_sys, human_mean_counts, type_key=0)

        # Record human average error counts (all) for each system in this domain
        for sys_name in gpt_by_sys.keys():
            vals = []
            for inst_id, h_counts in human_mean_counts.items():
                parts = inst_id.split("|")
                if len(parts) != 3:
                    continue
                dom_h, table_idx_str, human_sys = parts
                if dom_h != domain:
                    continue
                mapped_sys = HUMAN_SYS_MAP.get(domain, {}).get(human_sys)
                if mapped_sys != sys_name:
                    continue
                try:
                    table_idx = int(table_idx_str)
                except ValueError:
                    continue
                if table_idx not in idx_list:
                    continue
                vals.append(h_counts.get("all", 0.0))
            if vals:
                human_domain_avgs.setdefault(sys_name, {})[domain] = sum(vals) / len(vals)

    # Print a Table-4-style human avg error table (4 systems × 5 domains + NormAvg)
    if human_domain_avgs:
        print("\n" + "=" * 80)
        print("Human annotations: average #errors per output (all types), 4 systems × 5 domains")
        print("(values are averaged over three annotators; lower is better)")
        print("=" * 80)
        domains = sorted(indices_dict.keys())
        header = f"{'System':25s}" + "".join([f"{d:>12s}" for d in domains]) + f"{'NormAvg':>12s}"
        print(header)
        print("-" * len(header))

        # Compute per-domain min/max for normalization
        domain_minmax: Dict[str, Tuple[float, float]] = {}
        for d in domains:
            vals = [v for sys_vals in human_domain_avgs.values() if d in sys_vals for v in [sys_vals[d]]]
            if not vals:
                continue
            domain_minmax[d] = (min(vals), max(vals))

        def norm_val(domain: str, val: float) -> float:
            if domain not in domain_minmax:
                return 0.0
            mn, mx = domain_minmax[domain]
            if mx <= mn:
                return 0.0
            return (val - mn) / (mx - mn)

        for sys_name in ["primary", "zero_shot_1.7B", "sft_lora_1.7B", "ddkd_best"]:
            sys_vals = human_domain_avgs.get(sys_name, {})
            row_vals = []
            norm_list = []
            for d in domains:
                v = sys_vals.get(d)
                if v is None:
                    row_vals.append("   -   ")
                else:
                    row_vals.append(f"{v:8.2f}")
                    norm_list.append(norm_val(d, v))
            norm_avg = sum(norm_list) / len(norm_list) if norm_list else 0.0
            print(f"{sys_name:25s}" + "".join([f"{rv:>12s}" for rv in row_vals]) + f"{norm_avg:12.2f}")



if __name__ == "__main__":
    main()


