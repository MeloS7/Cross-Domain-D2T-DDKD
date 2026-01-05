#!/usr/bin/env python3
"""
Compute agreement between two LLM judges (GPT-5.1 and Gemini-2.5-pro)
for QUINTD-style datasets (Wikidata, weather, OWID, gsmarena, ice_hockey, ...).

For a given dataset, this script:
  - Discovers 14 system variants:
      * zero_shot:       Qwen3-{1.7B, 8B, 32B}
      * sft_lora:        Qwen3-{1.7B, 8B, 32B} with optional on_webnlg_json/csv/mkd suffixes
      * ddkd_zero_shot:  Qwen3-1.7B with {base, sub, pert, mixed}
      * ddkd_sft_lora:   Qwen3-1.7B with {base, sub, pert, mixed}
  - For each system, sample, and error type, builds binary vectors for GPT-5.1 and
    Gemini-2.5-pro.
  - Computes Pearson correlations at three levels:
      * word-level: concatenate token labels across systems and samples to get r_word;
      * example-level: concatenate per-sample error counts to get r_example;
      * system-level: average errors per sample per system to get r_system.

Directory layout (run from inner project root `Cross-Domain-D2T-DDKD/`):

  data/model_outputs/quintd/{dataset}/
    ├── outputs/
    │   ├── zero_shot/
    │   ├── sft_lora/
    │   ├── ddkd_zero_shot/
    │   └── ddkd_sft_lora/
    └── eval_res/
        ├── zero_shot/
        ├── sft_lora/
        ├── ddkd_zero_shot/
        └── ddkd_sft_lora/

Naming under eval_res (one file per judge):

  {dataset_name}_{method_name}_qwen3_{1.7b/8b/32b}{mid}_{gpt-5.1|gemini-2.5-pro}.jsonl

  - method_name ∈ {zero_shot, sft_lora, ddkd_zero_shot, ddkd_sft_lora}
  - mid:
      * zero_shot:            "" (no middle suffix)
      * sft_lora:             `_on_webnlg_json` / `_on_csv` / `_on_mkd` ...
      * ddkd_{...}:           `_base` / `_sub` / `_pert` / `_mixed`
  - Size suffix accepts `1.7b` or `1.7B`.

Naming for outputs (model generations):

  data/model_outputs/quintd/{dataset}/outputs/{method_name}/
      {dataset_name}_{method_name}_qwen3_{size}{mid}_responses.jsonl

Each eval_res JSONL line:
  - annotator_id
  - dataset
  - model
  - table_idx  (0..N-1)
  - annotations: list[dict], each error has:
      - reason: str
      - text:   str (error span)
      - type:   int in {0,1,2,3}
      - start:  int (character offset in the generated text)

Each outputs JSONL line:
  - response: str (the generated text being judged)
"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass
class SystemConfig:
    """Configuration for one evaluated system/model variant."""

    dataset: str
    method: str          # zero_shot / sft_lora / ddkd_zero_shot / ddkd_sft_lora
    size: str            # e.g. 1.7B / 8B / 32B
    variant: str         # "", on_webnlg_json/csv/mkd, base/sub/pert/mixed, ...
    name: str            # human-readable system name
    outputs_path: str
    judge_a_path: str    # GPT-5.1
    judge_b_path: str    # Gemini-2.5-pro


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_system_configs(base_dir: str, dataset: str) -> List[SystemConfig]:
    """
    Auto-discover the 14 system configurations for a given dataset.

    - Scan data/model_outputs/quintd/{dataset}/eval_res/ for
      {dataset}_{method}_qwen3_{size}{mid}_{judge}.jsonl
    - Group by (method, size, mid); require both gpt-5.1 and gemini-2.5-pro
    - Derive outputs path via the same base name:
        data/model_outputs/quintd/{dataset}/outputs/{method}/
            {dataset}_{method}_qwen3_{size}{mid}_responses.jsonl
    """

    eval_root = os.path.join(
        base_dir, "data", "model_outputs", "quintd", dataset, "eval_res"
    )
    out_root = os.path.join(
        base_dir, "data", "model_outputs", "quintd", dataset, "outputs"
    )

    if not os.path.isdir(eval_root):
        raise ValueError(f"Eval directory not found: {eval_root}")

    systems: List[SystemConfig] = []

    methods = ["zero_shot", "sft_lora", "ddkd_zero_shot", "ddkd_sft_lora"]

    from collections import defaultdict

    # key -> {"gpt-5.1": path, "gemini-2.5-pro": path}
    grouped: Dict[Tuple[str, str, str], Dict[str, str]] = defaultdict(dict)

    # Support two mid placements:
    #  1) {dataset}_{method}_qwen3_{size}{mid}_{judge}.jsonl
    #  2) {dataset}_{method}{mid}_qwen3_{size}_{judge}.jsonl
    #  Also accept gemini2.5-pro or gemini-2.5-pro spelling.
    pattern = re.compile(
        rf"^{re.escape(dataset)}_"
        r"(?P<method>zero_shot|sft_lora|ddkd_zero_shot|ddkd_sft_lora)"
        r"(?P<mid1>.*?)"              # mid1: may appear right after method
        r"_qwen3_(?P<size>[\d\.]+[bB])"
        r"(?P<mid2>.*?)_"             # mid2: may also appear after size
        r"(?P<judge>gpt-5\.1|gemini-?2\.5-pro)\.jsonl$"
    )

    for method in methods:
        subdir = os.path.join(eval_root, method)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if not fname.endswith(".jsonl"):
                continue
            m = pattern.match(fname)
            if not m:
                continue
            m_method = m.group("method")
            if m_method != method:
                continue
            size = m.group("size")
            mid1 = m.group("mid1") or ""
            mid2 = m.group("mid2") or ""
            mid = (mid1 + mid2) or ""
            judge = m.group("judge")
            size_norm = size.upper()

            # Keep only 1.7B for ddkd variants; skip 8B/32B
            if method in ("ddkd_zero_shot", "ddkd_sft_lora") and size_norm != "1.7B":
                continue

            key = (method, size_norm, mid)
            grouped[key][judge] = os.path.join(subdir, fname)

    if not grouped:
        raise ValueError(f"No eval_res files found under {eval_root} for dataset={dataset}")

    for (method, size, mid), judges in grouped.items():
        if "gpt-5.1" not in judges or "gemini-2.5-pro" not in judges:
            print(
                f"[Warning] For dataset={dataset}, system "
                f"(method={method}, size={size}, mid='{mid}') is missing "
                f"one of the judges: found={list(judges.keys())}, skip."
            )
            continue

        variant = (mid.lstrip("_") if mid else "")

        # Two common base_name arrangements:
        #   1) dataset_method_mid_qwen3_size   (e.g., sft_lora_on_webnlg_mkd_qwen3_32B)
        #   2) dataset_method_qwen3_size_mid   (e.g., ddkd_zero_shot_qwen3_1.7B_sub)
        base_name_1 = f"{dataset}_{method}{mid}_qwen3_{size}"
        base_name_2 = f"{dataset}_{method}_qwen3_{size}{mid}"

        # Prefer outputs under data/model_outputs/quintd/{dataset}/outputs/{method}/
        candidates = []
        outputs_dir_std = os.path.join(out_root, method)
        candidates.append(
            os.path.join(outputs_dir_std, base_name_1 + "_responses.jsonl")
        )
        candidates.append(
            os.path.join(outputs_dir_std, base_name_2 + "_responses.jsonl")
        )
        # Fallback: outputs file might be co-located with eval_res
        eval_dir = os.path.dirname(judges["gpt-5.1"])
        candidates.append(os.path.join(eval_dir, base_name_1 + "_responses.jsonl"))
        candidates.append(os.path.join(eval_dir, base_name_2 + "_responses.jsonl"))

        outputs_path: Optional[str] = None
        for cand in candidates:
            if os.path.isfile(cand):
                outputs_path = cand
                break

        if outputs_path is None:
            print(
                f"[Warning] Cannot find outputs (_responses.jsonl) for system "
                f"(dataset={dataset}, method={method}, size={size}, mid='{mid}').\n"
                f"  Tried: {candidates}\n"
                f"  -> Will skip word-level agreement for this system, "
                f"but still use it for example/system-level agreement."
            )

        if method == "zero_shot":
            sys_name = f"zero_shot_qwen3_{size}"
        elif method == "sft_lora":
            sys_name = f"sft_lora_qwen3_{size}_{variant or 'default'}"
        elif method == "ddkd_zero_shot":
            sys_name = f"ddkd_zero_shot_qwen3_{size}_{variant or 'base'}"
        else:  # ddkd_sft_lora
            sys_name = f"ddkd_sft_lora_qwen3_{size}_{variant or 'base'}"

        systems.append(
            SystemConfig(
                dataset=dataset,
                method=method,
                size=size,
                variant=variant,
                name=sys_name,
                outputs_path=outputs_path,
                judge_a_path=judges["gpt-5.1"],
                judge_b_path=judges["gemini-2.5-pro"],
            )
        )

    def sort_key(cfg: SystemConfig) -> Tuple[int, int, str]:
        method_order = {
            "zero_shot": 0,
            "sft_lora": 1,
            "ddkd_zero_shot": 2,
            "ddkd_sft_lora": 3,
        }.get(cfg.method, 99)
        size_order = {"1.7B": 0, "1.7b": 0, "8B": 1, "8b": 1, "32B": 2, "32b": 2}.get(
            cfg.size, 99
        )
        variant_order = {
            "": 0,
            "base": 0,
            "standard": 0,
            "sub": 1,
            "pert": 2,
            "mixed": 3,
        }.get(cfg.variant, 10)
        return (method_order, size_order, variant_order, cfg.variant)

    systems_sorted = sorted(systems, key=sort_key)

    print(
        f"Discovered {len(systems_sorted)} systems for dataset={dataset} "
        f"under {eval_root}"
    )

    return systems_sorted


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize text into whitespace-separated tokens, returning a list of
    (token, start_char, end_char) tuples.

    This uses a simple regex `\\S+` which should be consistent with how
    character offsets were computed (Python's str.find on the same text).
    """
    tokens: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        start = m.start()
        end = m.end()
        tokens.append((tok, start, end))
    return tokens


def build_token_labels(
    text: str,
    annotations: List[dict],
    target_type: Optional[int],
) -> List[int]:
    """
    Build a binary vector over tokens for a given text and annotation list.

    target_type:
      - None  => use all error types (\"all categories\")
      - 0/1/2/3 => use only errors with that 'type'
    """
    tokens = tokenize_with_spans(text)
    labels = [0] * len(tokens)

    for err in annotations:
        # filter by type
        if target_type is not None and err.get("type") != target_type:
            continue

        err_text = err.get("text", "")
        # Skip spans that cannot be matched in the original text (e.g., <OMISSION> or empty)
        if not isinstance(err_text, str):
            continue
        err_text_stripped = err_text.strip()
        if not err_text_stripped or err_text_stripped == "<OMISSION>":
            continue

        # Match error["text"] as a substring of the output, then mark overlapping tokens
        pattern = re.escape(err_text_stripped)
        for m in re.finditer(pattern, text):
            start = m.start()
            end = m.end()
            # mark all tokens that overlap [start, end)
            for i, (_tok, s, e) in enumerate(tokens):
                if s < end and e > start:
                    labels[i] = 1

    return labels


def pearson_corr(x: List[int], y: List[int]) -> float:
    """
    Compute Pearson correlation between two 0/1 vectors.
    Returns NaN if correlation is undefined (e.g. zero variance).
    """
    if len(x) == 0 or len(x) != len(y):
        return float("nan")

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

    if denom_x == 0 or denom_y == 0:
        return float("nan")

    return num / math.sqrt(denom_x * denom_y)


def compute_word_level_agreement(
    systems: List[SystemConfig],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute word-level Pearson correlations for each error category across all systems.

    Returns a dict mapping category name -> correlation.
    """
    # category_name -> (target_type or None for 'all')
    categories: Dict[str, Optional[int]] = {
        "all": None,
        "type0_incorrect_fact": 0,
        "type1_not_checkable": 1,
        "type2_misleading": 2,
        "type3_other": 3,
    }

    results: Dict[str, float] = {}

    for cat_name, cat_type in categories.items():
        all_w_a: List[int] = []
        all_w_b: List[int] = []

        if verbose:
            print(f"\n=== Computing word-level agreement for category: {cat_name} ===")

        for sys_cfg in systems:
            if verbose:
                print(f"  System: {sys_cfg.name}")

            # If outputs (_responses) file is missing, skip word-level for this system
            if not sys_cfg.outputs_path or not os.path.isfile(sys_cfg.outputs_path):
                if verbose:
                    print(
                        f"    [skip word-level] outputs file not found for {sys_cfg.name}: "
                        f"{sys_cfg.outputs_path}"
                    )
                continue

            outputs = load_jsonl(sys_cfg.outputs_path)
            judge_a = load_jsonl(sys_cfg.judge_a_path)
            judge_b = load_jsonl(sys_cfg.judge_b_path)

            if not (len(outputs) == len(judge_a) == len(judge_b)):
                raise ValueError(
                    f"Length mismatch for system {sys_cfg.name}: "
                    f"outputs={len(outputs)}, gpt-5.1={len(judge_a)}, gemini={len(judge_b)}"
                )

            # Build index by table_idx for safety
            ann_a_by_idx: Dict[int, dict] = {rec["table_idx"]: rec for rec in judge_a}
            ann_b_by_idx: Dict[int, dict] = {rec["table_idx"]: rec for rec in judge_b}

            for idx, out in enumerate(outputs):
                text = out.get("response", "")

                rec_a = ann_a_by_idx.get(idx)
                rec_b = ann_b_by_idx.get(idx)
                if rec_a is None or rec_b is None:
                    raise ValueError(
                        f"Missing annotation for table_idx={idx} in system {sys_cfg.name}"
                    )

                labels_a = build_token_labels(
                    text=text,
                    annotations=rec_a.get("annotations", []),
                    target_type=cat_type,
                )
                labels_b = build_token_labels(
                    text=text,
                    annotations=rec_b.get("annotations", []),
                    target_type=cat_type,
                )

                if len(labels_a) != len(labels_b):
                    raise ValueError(
                        f"Tokenization mismatch for system {sys_cfg.name}, "
                        f"example {idx}: len(a)={len(labels_a)}, len(b)={len(labels_b)}"
                    )

                all_w_a.extend(labels_a)
                all_w_b.extend(labels_b)

        r = pearson_corr(all_w_a, all_w_b)
        results[cat_name] = r

        if verbose:
            print(
                f"  -> Category {cat_name}: "
                f"n_tokens={len(all_w_a)}, r_word={r:.4f}"
            )

    return results


def compute_example_level_agreement(
    systems: List[SystemConfig],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Compute example-level Pearson correlations for each error category across all systems.

    For each example i (over all systems), for each judge j we compute:
        e_i^(j) = number of errors of the given category in example i
    and then correlate the concatenated vectors e^(A), e^(B).
    """
    categories: Dict[str, Optional[int]] = {
        "all": None,
        "type0_incorrect_fact": 0,
        "type1_not_checkable": 1,
        "type2_misleading": 2,
        "type3_other": 3,
    }

    results: Dict[str, float] = {}

    def count_errors(annotations: List[dict], target_type: Optional[int]) -> int:
        """Count errors of a given type in one example."""
        count = 0
        for err in annotations:
            t = err.get("type")
            # Only consider types 0–3; ignore special markers (e.g., -1: PARSING_FAILED)
            if t not in {0, 1, 2, 3}:
                continue
            if target_type is None or t == target_type:
                count += 1
        return count

    for cat_name, cat_type in categories.items():
        all_e_a: List[int] = []
        all_e_b: List[int] = []

        if verbose:
            print(f"\n=== Computing example-level agreement for category: {cat_name} ===")

        for sys_cfg in systems:
            if verbose:
                print(f"  System: {sys_cfg.name}")

            # Example-level only needs eval_res; outputs are not required
            judge_a = load_jsonl(sys_cfg.judge_a_path)
            judge_b = load_jsonl(sys_cfg.judge_b_path)

            if len(judge_a) != len(judge_b):
                raise ValueError(
                    f"[example-level] Length mismatch for system {sys_cfg.name}: "
                    f"gpt-5.1={len(judge_a)}, gemini={len(judge_b)}"
                )

            ann_a_by_idx: Dict[int, dict] = {rec["table_idx"]: rec for rec in judge_a}
            ann_b_by_idx: Dict[int, dict] = {rec["table_idx"]: rec for rec in judge_b}

            for idx in range(len(judge_a)):
                rec_a = ann_a_by_idx.get(idx)
                rec_b = ann_b_by_idx.get(idx)
                if rec_a is None or rec_b is None:
                    raise ValueError(
                        f"[example-level] Missing annotation for table_idx={idx} "
                        f"in system {sys_cfg.name}"
                    )

                e_a = count_errors(rec_a.get("annotations", []), cat_type)
                e_b = count_errors(rec_b.get("annotations", []), cat_type)

                all_e_a.append(e_a)
                all_e_b.append(e_b)

        r = pearson_corr(all_e_a, all_e_b)
        results[cat_name] = r

        if verbose:
            print(
                f"  -> Category {cat_name}: "
                f"n_examples={len(all_e_a)}, r_example={r:.4f}"
            )

    return results


def compute_system_level_agreement(
    systems: List[SystemConfig],
    verbose: bool = True,
) -> Tuple[
    Dict[str, float],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, float]],
]:
    """
    Compute system-level Pearson correlations for each error category across systems,
    and also return per-system average error counts (for GPT-5.1) for table reporting.

    For each system s and judge j:
        m_s^(j) = (1 / N_s) * sum_{i in s} e_i^(j)
    where e_i^(j) is the number of errors of the given category in example i,
    and N_s is the number of examples for system s.

    Returns:
      - results: category name -> r_system
      - avg_errors_judge_a: system_name -> {category_name -> average errors per output}
        (for judge A = GPT-5.1), used to print a table like Table 5.
    """
    categories: Dict[str, Optional[int]] = {
        "all": None,
        "type0_incorrect_fact": 0,
        "type1_not_checkable": 1,
        "type2_misleading": 2,
        "type3_other": 3,
    }

    def count_errors(annotations: List[dict], target_type: Optional[int]) -> int:
        """Count errors of a given type in one example."""
        count = 0
        for err in annotations:
            t = err.get("type")
            if t not in {0, 1, 2, 3}:
                continue
            if target_type is None or t == target_type:
                count += 1
        return count

    # Aggregates per system and judge
    # sys_totals[sys.name]["A"|"B"][cat_name] = total error count over all examples
    # sys_counts[sys.name] = N_s
    sys_totals: Dict[str, Dict[str, Dict[str, int]]] = {}
    sys_counts: Dict[str, int] = {}

    # First pass: accumulate counts
    for sys_cfg in systems:
        if verbose:
            print(f"\n=== Aggregating system-level stats for system: {sys_cfg.name} ===")

        # System-level uses eval_res only; outputs are not needed
        judge_a = load_jsonl(sys_cfg.judge_a_path)
        judge_b = load_jsonl(sys_cfg.judge_b_path)

        if len(judge_a) != len(judge_b):
            raise ValueError(
                f"[system-level] Length mismatch for system {sys_cfg.name}: "
                f"gpt-5.1={len(judge_a)}, gemini={len(judge_b)}"
            )

        ann_a_by_idx: Dict[int, dict] = {rec["table_idx"]: rec for rec in judge_a}
        ann_b_by_idx: Dict[int, dict] = {rec["table_idx"]: rec for rec in judge_b}

        sys_counts[sys_cfg.name] = len(judge_a)
        sys_totals.setdefault(sys_cfg.name, {"A": {}, "B": {}})
        for who in ["A", "B"]:
            for cat_name in categories.keys():
                sys_totals[sys_cfg.name][who].setdefault(cat_name, 0)

        for idx in range(len(judge_a)):
            rec_a = ann_a_by_idx.get(idx)
            rec_b = ann_b_by_idx.get(idx)
            if rec_a is None or rec_b is None:
                raise ValueError(
                    f"[system-level] Missing annotation for table_idx={idx} "
                    f"in system {sys_cfg.name}"
                )

            for cat_name, cat_type in categories.items():
                ea = count_errors(rec_a.get("annotations", []), cat_type)
                eb = count_errors(rec_b.get("annotations", []), cat_type)
                sys_totals[sys_cfg.name]["A"][cat_name] += ea
                sys_totals[sys_cfg.name]["B"][cat_name] += eb

    # Now compute r_system per category, and average errors for both judges
    results: Dict[str, float] = {}
    avg_errors_judge_a: Dict[str, Dict[str, float]] = {
        sys_name: {} for sys_name in sys_totals.keys()
    }
    avg_errors_judge_b: Dict[str, Dict[str, float]] = {
        sys_name: {} for sys_name in sys_totals.keys()
    }

    for cat_name in categories.keys():
        m_a: List[float] = []
        m_b: List[float] = []

        for sys_name in sys_totals.keys():
            n = sys_counts[sys_name]
            if n == 0:
                continue
            total_a = sys_totals[sys_name]["A"][cat_name]
            total_b = sys_totals[sys_name]["B"][cat_name]
            avg_a = total_a / n
            avg_b = total_b / n
            m_a.append(avg_a)
            m_b.append(avg_b)
            avg_errors_judge_a[sys_name][cat_name] = avg_a
            avg_errors_judge_b[sys_name][cat_name] = avg_b

        r = pearson_corr(m_a, m_b)
        results[cat_name] = r

        if verbose:
            print(
                f"\nSystem-level agreement for category {cat_name}: "
                f"n_systems={len(m_a)}, r_system={r:.4f}"
            )

    return results, avg_errors_judge_a, avg_errors_judge_b


# ===== Helpers for aggregated per-system stats across multiple datasets =====
def _normalize_variant(cfg: SystemConfig) -> str:
    """Normalize variant so that on_webnlg_{csv/json/mkd} collapses to on_webnlg."""
    var = (cfg.variant or "").lower()
    if cfg.method == "sft_lora":
        if var.startswith("on_webnlg"):
            return "on_webnlg"
        return var or "default"
    if cfg.method in ("ddkd_zero_shot", "ddkd_sft_lora"):
        return var or "base"
    return var


def _normalized_sys_key(cfg: SystemConfig) -> str:
    size = cfg.size.upper()
    var_norm = _normalize_variant(cfg)
    return f"{cfg.method}|{size}|{var_norm}"


def _pretty_name_normalized(key: str) -> str:
    # key: method|size|variant
    method, size, variant = key.split("|", 2)
    if method == "zero_shot":
        base = f"QWEN3-{size}"
    elif method == "sft_lora":
        base = f"QWEN3-{size}-LoRA-{variant}"
    elif method == "ddkd_zero_shot":
        base = f"QWEN3-{size}-DDKD-ZS-{variant.upper()}"
    elif method == "ddkd_sft_lora":
        base = f"QWEN3-{size}-DDKD-SFT-{variant.upper()}"
    else:
        base = f"{method}-{size}-{variant}"
    return base


def _aggregate_error_totals(
    systems: List[SystemConfig],
    judge: str = "A",
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate total error counts and example counts per normalized system key.
    Returns: key -> {"total": {type0..type3, all}, "n": int}
    """
    assert judge in ("A", "B")
    agg: Dict[str, Dict[str, float]] = {}
    for cfg in systems:
        path = cfg.judge_a_path if judge == "A" else cfg.judge_b_path
        if not os.path.isfile(path):
            continue
        records = load_jsonl(path)
        key = _normalized_sys_key(cfg)
        entry = agg.setdefault(
            key,
            {"total": {"type0": 0, "type1": 0, "type2": 0, "type3": 0, "all": 0}, "n": 0},
        )
        for rec in records:
            anns = rec.get("annotations", [])
            counts = {"type0": 0, "type1": 0, "type2": 0, "type3": 0}
            for err in anns:
                t = err.get("type")
                if t in {0, 1, 2, 3}:
                    counts[f"type{t}"] += 1
            counts["all"] = sum(counts[f"type{i}"] for i in range(4))
            for k, v in counts.items():
                entry["total"][k] += v
            entry["n"] += 1
    return agg


def _aggregate_domain_avgs(
    systems: List[SystemConfig],
    judge: str = "A",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Return domain -> norm_key -> {type0,type1,type2,type3,all} average errors.
    Averages are computed over samples for each system within a domain.
    """
    assert judge in ("A", "B")
    domain_map: Dict[str, Dict[str, Dict[str, float]]] = {}
    domain_counts: Dict[str, Dict[str, int]] = {}
    for cfg in systems:
        path = cfg.judge_a_path if judge == "A" else cfg.judge_b_path
        if not os.path.isfile(path):
            continue
        records = load_jsonl(path)
        key = _normalized_sys_key(cfg)
        dom = cfg.dataset
        domain_map.setdefault(dom, {}).setdefault(
            key, {"type0": 0, "type1": 0, "type2": 0, "type3": 0, "all": 0}
        )
        domain_counts.setdefault(dom, {}).setdefault(key, 0)
        for rec in records:
            anns = rec.get("annotations", [])
            counts = {"type0": 0, "type1": 0, "type2": 0, "type3": 0}
            for err in anns:
                t = err.get("type")
                if t in {0, 1, 2, 3}:
                    counts[f"type{t}"] += 1
            counts["all"] = sum(counts[f"type{i}"] for i in range(4))
            for k, v in counts.items():
                domain_map[dom][key][k] += v
            domain_counts[dom][key] += 1
    # Convert sums to averages
    for dom, sys_map in domain_map.items():
        for key, stats in sys_map.items():
            n = domain_counts[dom][key]
            if n == 0:
                continue
            for k in stats.keys():
                stats[k] /= n
    return domain_map


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute agreement between GPT-5.1 and Gemini-2.5-pro "
            "as LLM judges on Wikidata entity-description outputs "
            "(word-level and example-level)."
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
        "--dataset",
        type=str,
        help=(
            "Dataset name under data/model_outputs/quintd/, e.g. "
            "'wikidata', 'weather', 'owid', 'gsmarena', 'ice_hockey'. "
            "Ignored if --datasets is provided."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help=(
            "Comma-separated list of datasets to aggregate (e.g. "
            "wikidata,weather,owid,gsmarena,ice_hockey). "
            "If set, results will be computed on the union of all systems "
            "across the provided datasets."
        ),
    )
    parser.add_argument(
        "--no_verbose",
        action="store_true",
        help="Suppress per-system logging; only print final results.",
    )
    args = parser.parse_args()

    # Build systems for one or multiple datasets
    datasets = []
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise SystemExit("Please specify --dataset or --datasets")

    verbose = not args.no_verbose

    # If multiple datasets, aggregate systems then compute once; also print per-dataset if desired
    all_systems: List[SystemConfig] = []
    for ds in datasets:
        try:
            sys_list = build_system_configs(args.base_dir, ds)
        except Exception as e:
            print(f"[Error] building systems for dataset={ds}: {e}")
            continue
        if not sys_list:
            print(f"[Warning] No systems for dataset={ds}")
            continue
        if len(datasets) == 1:
            systems = sys_list
            word_results = compute_word_level_agreement(systems, verbose=verbose)
            example_results = compute_example_level_agreement(systems, verbose=verbose)
            (
                system_results,
                system_avgs_a,
                system_avgs_b,
            ) = compute_system_level_agreement(systems, verbose=verbose)
            print("\n" + "=" * 80)
            print(f"Results for dataset: {ds}")
            print("=" * 80)
            # Word-level
            print("\n" + "=" * 80)
            print("Word-level agreement between GPT-5.1 and Gemini-2.5-pro")
            print("=" * 80)
            print(f"{'Category':30s}  {'r_word':>10s}")
            print("-" * 80)
            for cat_name, r in word_results.items():
                r_str = f"{r:.4f}" if not math.isnan(r) else "NaN"
                print(f"{cat_name:30s}  {r_str:>10s}")
            print("=" * 80)
            # System-level
            print("\n" + "=" * 80)
            print("System-level agreement between GPT-5.1 and Gemini-2.5-pro")
            print("=" * 80)
            print(f"{'Category':30s}  {'r_system':>10s}")
            print("-" * 80)
            for cat_name, r in system_results.items():
                r_str = f"{r:.4f}" if not math.isnan(r) else "NaN"
                print(f"{cat_name:30s}  {r_str:>10s}")
            print("=" * 80)
            # Pretty table of average errors per output per system (judge A)
            print("\n" + "=" * 80)
            print("Average number of errors per output per system (judge = GPT-5.1)")
            print("=" * 80)
            header = (
                f"{'Model':35s}"
                f"{'Incorrect Fact':>16s}"
                f"{'Not Checkable':>16s}"
                f"{'Misleading':>12s}"
                f"{'Other':>10s}"
                f"{'All Categories':>16s}"
            )
            print(header)
            print("-" * len(header))
            cfg_by_name: Dict[str, SystemConfig] = {cfg.name: cfg for cfg in systems}

            def group_systems(systems_list: List[SystemConfig]) -> List[Tuple[str, List[str]]]:
                method_to_group_name = {
                    "zero_shot": "Zero-Shot",
                    "sft_lora": "SFT LoRA",
                    "ddkd_zero_shot": "DDKD from Zero-Shot",
                    "ddkd_sft_lora": "DDKD from SFT LoRA",
                }

                groups_map: Dict[str, List[SystemConfig]] = {}
                for cfg in systems_list:
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
                        "sub": 1,
                        "pert": 2,
                        "mixed": 3,
                    }.get(cfg.variant, 10)
                    return (size_order, variant_order, cfg.variant)

                groups_list: List[Tuple[str, List[str]]] = []
                for group_name, cfgs in groups_map.items():
                    cfgs_sorted = sorted(cfgs, key=sort_key)
                    groups_list.append((group_name, [c.name for c in cfgs_sorted]))

                return groups_list

            groups = group_systems(systems)

            def pretty_model_name(sys_name: str) -> str:
                cfg = cfg_by_name.get(sys_name)
                if cfg is None:
                    return sys_name
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
                return sys_name

            for group_name, sys_list in groups:
                print(f"\n# {group_name}")
                for sys_name in sys_list:
                    if sys_name not in system_avgs_a:
                        continue
                    stats = system_avgs_a[sys_name]
                    row = (
                        f"{pretty_model_name(sys_name):35s}"
                        f"{stats.get('type0_incorrect_fact', 0.0):16.2f}"
                        f"{stats.get('type1_not_checkable', 0.0):16.2f}"
                        f"{stats.get('type2_misleading', 0.0):12.2f}"
                        f"{stats.get('type3_other', 0.0):10.2f}"
                        f"{stats.get('all', 0.0):16.2f}"
                    )
                    print(row)

            print("=" * 80)

            # Gemini averages table
            print("\n" + "=" * 80)
            print("Average number of errors per output per system (judge = Gemini-2.5-pro)")
            print("=" * 80)
            print(header)
            print("-" * len(header))
            for group_name, sys_list in groups:
                print(f"\n# {group_name}")
                for sys_name in sys_list:
                    if sys_name not in system_avgs_b:
                        continue
                    stats = system_avgs_b[sys_name]
                    row = (
                        f"{pretty_model_name(sys_name):35s}"
                        f"{stats.get('type0_incorrect_fact', 0.0):16.2f}"
                        f"{stats.get('type1_not_checkable', 0.0):16.2f}"
                        f"{stats.get('type2_misleading', 0.0):12.2f}"
                        f"{stats.get('type3_other', 0.0):10.2f}"
                        f"{stats.get('all', 0.0):16.2f}"
                    )
                    print(row)

            print("=" * 80)

            print("\n" + "=" * 80)
            print("Example-level agreement between GPT-5.1 and Gemini-2.5-pro")
            print("=" * 80)
            print(f"{'Category':30s}  {'r_example':>10s}")
            print("-" * 80)
            for cat_name, r in example_results.items():
                r_str = f"{r:.4f}" if not math.isnan(r) else "NaN"
                print(f"{cat_name:30s}  {r_str:>10s}")
            print("=" * 80)

        all_systems.extend(sys_list)

    # Aggregated over all provided datasets
    if len(datasets) > 1:
        if not all_systems:
            print("[Warning] No systems aggregated across datasets.")
            return
        print("\n" + "=" * 80)
        print(f"Aggregated results across datasets: {', '.join(datasets)}")
        print("=" * 80)
        word_results = compute_word_level_agreement(all_systems, verbose=False)
        example_results = compute_example_level_agreement(all_systems, verbose=False)
        system_results, system_avgs_a_all, system_avgs_b_all = compute_system_level_agreement(
            all_systems, verbose=False
        )

        # Re-aggregate: merge systems sharing method+size+normalized variant, then average across domains
        agg_a = _aggregate_error_totals(all_systems, judge="A")
        agg_b = _aggregate_error_totals(all_systems, judge="B")
        dom_avgs_a = _aggregate_domain_avgs(all_systems, judge="A")
        dom_avgs_b = _aggregate_domain_avgs(all_systems, judge="B")

        # Additionally load GPT-4.1 (GPT-5.1 judge only, path zero_shot/{domain}_zero_shot_gpt4.1_gpt-5.1.jsonl)
        base_path = Path(args.base_dir)

        def load_gpt41(domain: str) -> List[dict]:
            path = base_path / f"data/model_outputs/quintd/{domain}/eval_res/zero_shot/{domain}_zero_shot_gpt4.1_gpt-5.1.jsonl"
            if not path.is_file():
                return []
            return load_jsonl(str(path))

        gpt41_key = "zero_shot|GPT4.1|base"
        gpt41_dom_avgs: Dict[str, Dict[str, Dict[str, float]]] = {}
        gpt41_agg: Dict[str, Dict[str, float]] = {gpt41_key: {"total": {"type0": 0, "type1": 0, "type2": 0, "type3": 0, "all": 0}, "n": 0}}

        for dom in datasets:
            recs = load_gpt41(dom)
            if not recs:
                continue
            stats_sum = {"type0": 0, "type1": 0, "type2": 0, "type3": 0, "all": 0}
            for rec in recs:
                counts = {"type0": 0, "type1": 0, "type2": 0, "type3": 0}
                for err in rec.get("annotations", []):
                    t = err.get("type")
                    if t in {0, 1, 2, 3}:
                        counts[f"type{t}"] += 1
                counts["all"] = sum(counts[f"type{i}"] for i in range(4))
                for k, v in counts.items():
                    stats_sum[k] += v
                gpt41_agg[gpt41_key]["total"][k] = gpt41_agg[gpt41_key]["total"].get(k, 0) + v
                gpt41_agg[gpt41_key]["n"] += 1
            n = len(recs)
            if n > 0:
                gpt41_dom_avgs.setdefault(dom, {})[gpt41_key] = {k: stats_sum[k] / n for k in stats_sum.keys()}
        # Merge GPT-4.1 stats into aggregated tables (judge = GPT-5.1 only)
        if gpt41_dom_avgs:
            for dom, sys_map in gpt41_dom_avgs.items():
                dom_avgs_a.setdefault(dom, {}).update(sys_map)
            agg_a.update(gpt41_agg)

        print("\n[Aggregated] Word-level agreement")
        for cat_name, r in word_results.items():
            r_str = f"{r:.4f}" if not math.isnan(r) else "NaN"
            print(f"  {cat_name:30s}  {r_str:>10s}")

        print("\n[Aggregated] Example-level agreement")
        for cat_name, r in example_results.items():
            r_str = f"{r:.4f}" if not math.isnan(r) else "NaN"
            print(f"  {cat_name:30s}  {r_str:>10s}")

        print("\n[Aggregated] System-level agreement")
        for cat_name, r in system_results.items():
            r_str = f"{r:.4f}" if not math.isnan(r) else "NaN"
            print(f"  {cat_name:30s}  {r_str:>10s}")

        # Print cross-domain per-system average error tables (GPT-5.1 / Gemini-2.5-pro)
        print("\n" + "=" * 80)
        print("Aggregated across all datasets: average errors per output per system (judge = GPT-5.1)")
        print("=" * 80)
        header = (
            f"{'System':40s}"
            f"{'Incorrect Fact':>16s}"
            f"{'Not Checkable':>16s}"
            f"{'Misleading':>12s}"
            f"{'Other':>10s}"
            f"{'All Categories':>16s}"
            f"{'NormAll':>12s}"
        )
        print(header)
        print("-" * len(header))

        # Compute per-domain min-max for normalization
        def build_minmax(dom_avgs: Dict[str, Dict[str, Dict[str, float]]]):
            mm: Dict[str, Dict[str, tuple]] = {}
            for dom, sys_map in dom_avgs.items():
                mm.setdefault(dom, {})
                for k in ["type0", "type1", "type2", "type3", "all"]:
                    vals = [stats[k] for stats in sys_map.values() if stats]
                    if not vals:
                        continue
                    mm[dom][k] = (min(vals), max(vals))
            return mm

        mm_a = build_minmax(dom_avgs_a)

        def macro_avg(dom_avgs: Dict[str, Dict[str, Dict[str, float]]], key: str) -> Dict[str, float]:
            """Macro average: first average within each domain, then across domains (no normalization)."""
            vals: Dict[str, List[float]] = {k: [] for k in ["type0", "type1", "type2", "type3", "all"]}
            for dom, sys_map in dom_avgs.items():
                stats = sys_map.get(key)
                if not stats:
                    continue
                for k in vals.keys():
                    vals[k].append(stats.get(k, 0.0))
            return {k: (sum(v) / len(v) if v else 0.0) for k, v in vals.items()}

        def norm_avg(dom_avgs: Dict[str, Dict[str, Dict[str, float]]], mm: Dict[str, Dict[str, tuple]]):
            norm_res: Dict[str, Dict[str, float]] = {}
            for dom, sys_map in dom_avgs.items():
                for key, stats in sys_map.items():
                    for k in ["type0", "type1", "type2", "type3", "all"]:
                        mn, mx = mm.get(dom, {}).get(k, (None, None))
                        if mn is None or mx is None or mx <= mn:
                            continue
                        val = (stats[k] - mn) / (mx - mn)
                        norm_res.setdefault(key, {}).setdefault(k, []).append(val)
            # Average across domains
            norm_avg_res: Dict[str, Dict[str, float]] = {}
            for key, type_map in norm_res.items():
                norm_avg_res[key] = {k: (sum(v) / len(v) if v else 0.0) for k, v in type_map.items()}
            return norm_avg_res

        norm_a = norm_avg(dom_avgs_a, mm_a)

        for key in sorted(agg_a.keys()):
            stats = macro_avg(dom_avgs_a, key)
            norm_all = norm_a.get(key, {}).get("all", 0.0)
            row = (
                f"{_pretty_name_normalized(key):40s}"
                f"{stats.get('type0', 0.0):16.2f}"
                f"{stats.get('type1', 0.0):16.2f}"
                f"{stats.get('type2', 0.0):12.2f}"
                f"{stats.get('type3', 0.0):10.2f}"
                f"{stats.get('all', 0.0):16.2f}"
                f"{norm_all:12.2f}"
            )
            print(row)

        print("=" * 80)

        print("\n" + "=" * 80)
        print("Aggregated across all datasets: average errors per output per system (judge = Gemini-2.5-pro)")
        print("=" * 80)
        print(header)
        print("-" * len(header))
        mm_b = build_minmax(dom_avgs_b)
        norm_b = norm_avg(dom_avgs_b, mm_b)
        for key in sorted(agg_b.keys()):
            stats = macro_avg(dom_avgs_b, key)
            norm_all = norm_b.get(key, {}).get("all", 0.0)
            row = (
                f"{_pretty_name_normalized(key):40s}"
                f"{stats.get('type0', 0.0):16.2f}"
                f"{stats.get('type1', 0.0):16.2f}"
                f"{stats.get('type2', 0.0):12.2f}"
                f"{stats.get('type3', 0.0):10.2f}"
                f"{stats.get('all', 0.0):16.2f}"
                f"{norm_all:12.2f}"
            )
            print(row)

        print("=" * 80)

    # Only print per-system tables when a single dataset is evaluated
    if len(datasets) == 1:
        print("\n" + "=" * 80)
        print("Word-level agreement between GPT-5.1 and Gemini-2.5-pro")
        print("=" * 80)
        print(f"{'Category':30s}  {'r_word':>10s}")
        print("-" * 80)
        for cat_name, r in word_results.items():
            r_str = f"{r:.4f}" if not math.isnan(r) else "NaN"
            print(f"{cat_name:30s}  {r_str:>10s}")
        print("=" * 80)

        print("\n" + "=" * 80)
        print("System-level agreement between GPT-5.1 and Gemini-2.5-pro")
        print("=" * 80)
        print(f"{'Category':30s}  {'r_system':>10s}")
        print("-" * 80)
        for cat_name, r in system_results.items():
            r_str = f"{r:.4f}" if not math.isnan(r) else "NaN"
            print(f"{cat_name:30s}  {r_str:>10s}")
        print("=" * 80)

        # Pretty table of average errors per output per system (judge A)
        print("\n" + "=" * 80)
        print("Average number of errors per output per system (judge = GPT-5.1)")
        print("=" * 80)
        header = (
            f"{'Model':35s}"
            f"{'Incorrect Fact':>16s}"
            f"{'Not Checkable':>16s}"
            f"{'Misleading':>12s}"
            f"{'Other':>10s}"
            f"{'All Categories':>16s}"
        )
        print(header)
        print("-" * len(header))

        # Group systems by method inferred from `systems`
        cfg_by_name: Dict[str, SystemConfig] = {cfg.name: cfg for cfg in systems}

        def group_systems(
            systems_list: List[SystemConfig],
        ) -> List[Tuple[str, List[str]]]:
            method_to_group_name = {
                "zero_shot": "Zero-Shot",
                "sft_lora": "SFT LoRA",
                "ddkd_zero_shot": "DDKD from Zero-Shot",
                "ddkd_sft_lora": "DDKD from SFT LoRA",
            }

            groups_map: Dict[str, List[SystemConfig]] = {}
            for cfg in systems_list:
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
                    "sub": 1,
                    "pert": 2,
                    "mixed": 3,
                }.get(cfg.variant, 10)
                return (size_order, variant_order, cfg.variant)

            groups_list: List[Tuple[str, List[str]]] = []
            for group_name, cfgs in groups_map.items():
                cfgs_sorted = sorted(cfgs, key=sort_key)
                groups_list.append((group_name, [c.name for c in cfgs_sorted]))

            return groups_list

        groups = group_systems(systems)

        def pretty_model_name(sys_name: str) -> str:
            cfg = cfg_by_name.get(sys_name)
            if cfg is None:
                return sys_name
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
            return sys_name

        for group_name, sys_list in groups:
            print(f"\n# {group_name}")
            for sys_name in sys_list:
                if sys_name not in system_avgs_a:
                    continue
                stats = system_avgs_a[sys_name]
                row = (
                    f"{pretty_model_name(sys_name):35s}"
                    f"{stats.get('type0_incorrect_fact', 0.0):16.2f}"
                    f"{stats.get('type1_not_checkable', 0.0):16.2f}"
                    f"{stats.get('type2_misleading', 0.0):12.2f}"
                    f"{stats.get('type3_other', 0.0):10.2f}"
                    f"{stats.get('all', 0.0):16.2f}"
                )
                print(row)

        print("=" * 80)

        # Pretty table for Gemini-2.5-pro (judge B)
        print("\n" + "=" * 80)
        print("Average number of errors per output per system (judge = Gemini-2.5-pro)")
        print("=" * 80)
        print(header)
        print("-" * len(header))

        for group_name, sys_list in groups:
            print(f"\n# {group_name}")
            for sys_name in sys_list:
                if sys_name not in system_avgs_b:
                    continue
                stats = system_avgs_b[sys_name]
                row = (
                    f"{pretty_model_name(sys_name):35s}"
                    f"{stats.get('type0_incorrect_fact', 0.0):16.2f}"
                    f"{stats.get('type1_not_checkable', 0.0):16.2f}"
                    f"{stats.get('type2_misleading', 0.0):12.2f}"
                    f"{stats.get('type3_other', 0.0):10.2f}"
                    f"{stats.get('all', 0.0):16.2f}"
                )
                print(row)

        print("=" * 80)

        print("\n" + "=" * 80)
        print("Example-level agreement between GPT-5.1 and Gemini-2.5-pro")
        print("=" * 80)
        print(f"{'Category':30s}  {'r_example':>10s}")
        print("-" * 80)
        for cat_name, r in example_results.items():
            r_str = f"{r:.4f}" if not math.isnan(r) else "NaN"
            print(f"{cat_name:30s}  {r_str:>10s}")
        print("=" * 80)


if __name__ == "__main__":
    main()


