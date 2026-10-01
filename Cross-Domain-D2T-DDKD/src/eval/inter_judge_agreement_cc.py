#!/usr/bin/env python3
"""
Inter-judge agreement (GPT-5.1 vs Gemini-2.5-Pro) for Content Coverage (CC) evaluation.

Computes:
  - Instance-level: Krippendorff's alpha on paired instances across all matched systems.
      * coverage_score (1–5, ordinal)        → α-ordinal metric
      * estimated_coverage_ratio (0–1, interval) → α-interval metric
    Instances where either judge has parse_error are skipped.

  - System-level: Pearson r and Spearman ρ between per-system mean scores.
    System means are computed excluding parse_error instances, then correlated.
    Spearman ρ (rank-based) is particularly informative here because we care
    about whether the two judges agree on system ranking, not just magnitude.

Reports per-domain and cross-domain (pooled) results in ASCII tables.

Usage from the project root:
  python src/data_analysis/inter_judge_agreement_cc.py
  python src/data_analysis/inter_judge_agreement_cc.py --data_dir data/model_outputs/quintd
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy import stats as _sp


DOMAINS = ["owid", "gsmarena", "weather", "ice_hockey", "wikidata"]
JUDGE_A = "gpt5.1"
JUDGE_B = "gemini2.5-pro"
METRICS = ["coverage_score", "estimated_coverage_ratio"]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def extract_judge_and_stem(path: Path) -> Tuple[str, str]:
    """
    `owid_zero_shot_qwen3_1.7b_cc_gpt5.1.jsonl`  →  judge='gpt5.1', stem='owid_zero_shot_qwen3_1.7b'
    """
    stem = path.stem
    if "_cc_" in stem:
        base, judge = stem.rsplit("_cc_", 1)
        return judge, base
    return "", stem


def discover_pairs(
    data_dir: Path,
) -> Dict[str, Dict[str, Dict[str, Path]]]:
    """
    Scan data_dir for *_cc_*.jsonl files.

    Returns:
        { domain: { system_key: { judge: Path } } }
    where system_key = "<method_subdir>/<stem_without_judge>".
    """
    result: Dict[str, Dict[str, Dict[str, Path]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for domain in DOMAINS:
        domain_dir = data_dir / domain
        if not domain_dir.exists():
            continue
        for p in sorted(domain_dir.rglob("*_cc_*.jsonl")):
            judge, stem = extract_judge_and_stem(p)
            if not judge:
                continue
            # method subdir sits right after eval_res/
            rel_parts = list(p.relative_to(domain_dir).parts)
            if "eval_res" in rel_parts:
                ei = rel_parts.index("eval_res")
                method = rel_parts[ei + 1] if ei + 1 < len(rel_parts) - 1 else ""
            else:
                method = ""
            system_key = f"{method}/{stem}" if method else stem
            result[domain][system_key][judge] = p
    return result


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def krippendorff_alpha_interval(
    values_a: List[float], values_b: List[float]
) -> Optional[float]:
    """
    Krippendorff's alpha for 2 raters, interval metric (d² = (v_i − v_j)²).
    Suitable for continuous measurements (e.g. estimated_coverage_ratio).

    alpha = 1 − D_o / D_e
    D_o  = (1/n) Σ_i (a_i − b_i)²
    D_e  = Σ_{c≠k}(c−k)² / (N(N−1))   [pooled values]
    Uses Σ_{i≠j}(x_i−x_j)² = 2N·Σx² − 2(Σx)²  for O(n) computation.
    """
    n = len(values_a)
    if n < 2:
        return None

    d_o = sum((a - b) ** 2 for a, b in zip(values_a, values_b)) / n

    all_vals = values_a + values_b
    N = len(all_vals)
    sum_v = sum(all_vals)
    sum_v2 = sum(v * v for v in all_vals)
    sum_sq_diffs = 2 * N * sum_v2 - 2 * sum_v ** 2
    d_e = sum_sq_diffs / (N * (N - 1))

    if d_e == 0:
        return 1.0 if d_o == 0 else None
    return 1.0 - d_o / d_e


def krippendorff_alpha_ordinal(
    values_a: List[int], values_b: List[int]
) -> Optional[float]:
    """
    Krippendorff's alpha for 2 raters, ordinal metric.
    Suitable for ordered categorical data (e.g. coverage_score 1–5).

    The ordinal difference function is:
        d²(c, k) = ( Σ_{g=min(c,k)}^{max(c,k)} n_g  −  (n_c + n_k) / 2 )²

    where n_g is the frequency of value g in the pooled distribution of all
    values from both raters. This weights disagreement by how many categories
    lie between c and k in the empirical distribution, rather than their raw
    numeric difference.

    alpha = 1 − D_o / D_e
    D_o = (1/n) Σ_i d²(a_i, b_i)
    D_e = Σ_{c≠k} d²(c, k) * n_c * n_k / (N * (N - 1))
    """
    n = len(values_a)
    if n < 2:
        return None

    # Build pooled frequency table from integer values
    all_vals = values_a + values_b
    categories = sorted(set(all_vals))
    freq: Dict[int, int] = {c: 0 for c in categories}
    for v in all_vals:
        freq[int(v)] += 1
    N = len(all_vals)

    def ordinal_d2(c: int, k: int) -> float:
        if c == k:
            return 0.0
        lo, hi = min(c, k), max(c, k)
        # Σ_{g=lo}^{hi} n_g  then subtract (n_lo + n_hi) / 2
        cumsum = sum(freq.get(g, 0) for g in range(lo, hi + 1))
        return (cumsum - (freq.get(lo, 0) + freq.get(hi, 0)) / 2.0) ** 2

    # Observed disagreement
    d_o = sum(ordinal_d2(int(a), int(b)) for a, b in zip(values_a, values_b)) / n

    # Expected disagreement
    d_e = sum(
        ordinal_d2(c, k) * freq[c] * freq[k]
        for c in categories
        for k in categories
        if c != k
    ) / (N * (N - 1))

    if d_e == 0:
        return 1.0 if d_o == 0 else None
    return 1.0 - d_o / d_e


# ---------------------------------------------------------------------------
# Correlation helpers (scipy-backed)
# ---------------------------------------------------------------------------

def pearson_rp(
    xs: List[float], ys: List[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Pearson r and two-tailed p-value via scipy.stats.pearsonr."""
    if len(xs) < 3:
        return None, None
    r, p = _sp.pearsonr(xs, ys)
    return float(r), float(p)


def spearman_rp(
    xs: List[float], ys: List[float]
) -> Tuple[Optional[float], Optional[float]]:
    """Spearman ρ and two-tailed p-value via scipy.stats.spearmanr.

    Handles ties correctly; p-value uses the t-approximation (adequate for
    pooled n; per-domain p-values are intentionally not displayed).
    """
    if len(xs) < 3:
        return None, None
    res = _sp.spearmanr(xs, ys)
    rho = float(getattr(res, "statistic", res.correlation))
    return rho, float(res.pvalue)


# ---------------------------------------------------------------------------
# Pair collection: instance-level
# ---------------------------------------------------------------------------

def collect_instance_pairs(
    domain_systems: Dict[str, Dict[str, Path]],
    metric: str,
) -> Tuple[List[float], List[float], int]:
    """
    Collect aligned instance-level (judge_a_val, judge_b_val) pairs across all
    systems in a domain. Skips instances with parse_error or missing values.

    Returns (values_a, values_b, n_skipped_parse_error)
    """
    vals_a: List[float] = []
    vals_b: List[float] = []
    n_skip = 0

    for system_key, judge_files in sorted(domain_systems.items()):
        if JUDGE_A not in judge_files or JUDGE_B not in judge_files:
            continue

        recs_a = {r["table_idx"]: r for r in load_jsonl(judge_files[JUDGE_A]) if "table_idx" in r}
        recs_b = {r["table_idx"]: r for r in load_jsonl(judge_files[JUDGE_B]) if "table_idx" in r}

        for idx in sorted(set(recs_a) & set(recs_b)):
            ra, rb = recs_a[idx], recs_b[idx]
            if ra.get("parse_error") or rb.get("parse_error"):
                n_skip += 1
                continue
            va = ra.get(metric)
            vb = rb.get(metric)
            if va is None or vb is None:
                n_skip += 1
                continue
            vals_a.append(float(va))
            vals_b.append(float(vb))

    return vals_a, vals_b, n_skip


# ---------------------------------------------------------------------------
# Pair collection: system-level
# ---------------------------------------------------------------------------

def collect_system_means(
    domain_systems: Dict[str, Dict[str, Path]],
    metric: str,
) -> Tuple[List[float], List[float], List[str]]:
    """
    For each system that has both judges, compute mean metric value per judge
    (excluding parse_error instances).

    Returns (means_a, means_b, system_labels)
    """
    means_a: List[float] = []
    means_b: List[float] = []
    labels: List[str] = []

    for system_key, judge_files in sorted(domain_systems.items()):
        if JUDGE_A not in judge_files or JUDGE_B not in judge_files:
            continue

        valid_a = [
            float(r[metric])
            for r in load_jsonl(judge_files[JUDGE_A])
            if not r.get("parse_error") and r.get(metric) is not None
        ]
        valid_b = [
            float(r[metric])
            for r in load_jsonl(judge_files[JUDGE_B])
            if not r.get("parse_error") and r.get(metric) is not None
        ]
        if not valid_a or not valid_b:
            continue

        means_a.append(statistics.mean(valid_a))
        means_b.append(statistics.mean(valid_b))
        labels.append(system_key)

    return means_a, means_b, labels


# ---------------------------------------------------------------------------
# ASCII table rendering
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], decimals: int = 3) -> str:
    if v is None:
        return "-"
    return f"{v:.{decimals}f}"


def _fmt_rp(r: Optional[float], p: Optional[float]) -> str:
    """Format a correlation with its p-value: '0.850 (p=.023)' or '0.850 (p<.001)'."""
    if r is None:
        return "-"
    r_str = f"{r:.3f}"
    if p is None:
        return r_str
    if p < 0.001:
        return f"{r_str} (p<.001)"
    return f"{r_str} (p={p:.3f})"


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def fmt_row(cells):
        return "|" + "|".join(f" {c:<{widths[i]}} " for i, c in enumerate(cells)) + "|"

    lines = [sep, fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inter-judge agreement (GPT-5.1 vs Gemini-2.5-Pro) for CC evaluation."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/model_outputs/quintd",
        help="Root directory containing per-domain model outputs.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    domain_systems = discover_pairs(data_dir)

    # ------------------------------------------------------------------ #
    # Per-domain computation                                               #
    # ------------------------------------------------------------------ #
    # instance-level: { domain: { metric: (alpha, n_pairs, n_skip) } }
    # system-level:   { domain: { metric: (pearson_r, n_systems,
    #                                       means_a, means_b) } }

    inst_results: Dict[str, Dict[str, tuple]] = {}
    sys_results: Dict[str, Dict[str, tuple]] = {}

    # Pooled across domains
    pooled_inst: Dict[str, Tuple[List[float], List[float]]] = {
        m: ([], []) for m in METRICS
    }
    pooled_sys: Dict[str, Tuple[List[float], List[float]]] = {
        m: ([], []) for m in METRICS
    }

    for domain in DOMAINS:
        if domain not in domain_systems:
            continue
        systems = domain_systems[domain]

        inst_results[domain] = {}
        sys_results[domain] = {}

        for metric in METRICS:
            # Instance-level
            vals_a, vals_b, n_skip = collect_instance_pairs(systems, metric)
            if metric == "coverage_score":
                alpha = krippendorff_alpha_ordinal(
                    [int(v) for v in vals_a], [int(v) for v in vals_b]
                )
            else:
                alpha = krippendorff_alpha_interval(vals_a, vals_b)
            inst_results[domain][metric] = (alpha, len(vals_a), n_skip)
            pooled_inst[metric][0].extend(vals_a)
            pooled_inst[metric][1].extend(vals_b)

            # System-level
            means_a, means_b, sys_labels = collect_system_means(systems, metric)
            n_sys = len(means_a)
            r_p, p_rp = pearson_rp(means_a, means_b)
            r_s, p_rs = spearman_rp(means_a, means_b)
            sys_results[domain][metric] = (r_p, r_s, p_rp, p_rs, n_sys, means_a, means_b, sys_labels)
            pooled_sys[metric][0].extend(means_a)
            pooled_sys[metric][1].extend(means_b)

    # ------------------------------------------------------------------ #
    # Cross-domain (pooled)                                                #
    # ------------------------------------------------------------------ #
    pooled_inst_summary: Dict[str, tuple] = {}
    pooled_sys_summary: Dict[str, tuple] = {}

    for metric in METRICS:
        va, vb = pooled_inst[metric]
        if metric == "coverage_score":
            alpha = krippendorff_alpha_ordinal(
                [int(v) for v in va], [int(v) for v in vb]
            )
        else:
            alpha = krippendorff_alpha_interval(va, vb)
        pooled_inst_summary[metric] = (alpha, len(va))

        ma, mb = pooled_sys[metric]
        n_pool = len(ma)
        r_p, p_rp = pearson_rp(ma, mb)
        r_s, p_rs = spearman_rp(ma, mb)
        pooled_sys_summary[metric] = (r_p, r_s, p_rp, p_rs, n_pool)

    # ------------------------------------------------------------------ #
    # Print: Instance-level table                                          #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("Instance-level Inter-Judge Agreement  (Krippendorff's α)")
    print("  coverage_score : α-ordinal  |  estimated_coverage_ratio : α-interval")
    print("Judges: {} vs {}".format(JUDGE_A, JUDGE_B))
    print("=" * 70)

    inst_headers = [
        "Domain", "α-ord (score)", "α-int (ratio)", "n_pairs (score)", "n_pairs (ratio)", "n_skip",
    ]
    inst_rows = []
    for domain in DOMAINS:
        if domain not in inst_results:
            continue
        score_alpha, n_s, skip_s = inst_results[domain]["coverage_score"]
        ratio_alpha, n_r, skip_r = inst_results[domain]["estimated_coverage_ratio"]
        inst_rows.append([
            domain,
            _fmt(score_alpha),
            _fmt(ratio_alpha),
            str(n_s),
            str(n_r),
            str(skip_s),  # skip_s == skip_r in practice (same logic)
        ])
    # Pooled row
    a_s, n_s = pooled_inst_summary["coverage_score"]
    a_r, n_r = pooled_inst_summary["estimated_coverage_ratio"]
    inst_rows.append([
        "ALL (pooled)", _fmt(a_s), _fmt(a_r), str(n_s), str(n_r), "-",
    ])
    print(render_table(inst_headers, inst_rows))

    # ------------------------------------------------------------------ #
    # Print: System-level table                                            #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("System-level Inter-Judge Agreement  (Pearson r  &  Spearman ρ)")
    print("  Per-domain: r/ρ only (n is typically small, p-values have low power).")
    print("  ALL (pooled): r/ρ with two-tailed p  [t = corr*sqrt(n-2)/sqrt(1-corr²), df=n-2]")
    print("  Note: p for Spearman ρ uses the same t-approximation (valid for pooled n).")
    print("Judges: {} vs {}".format(JUDGE_A, JUDGE_B))
    print("=" * 80)

    sys_headers = [
        "Domain", "r (score)", "ρ (score)", "r (ratio)", "ρ (ratio)", "n_sys",
    ]
    sys_rows = []
    for domain in DOMAINS:
        if domain not in sys_results:
            continue
        rp_s, rs_s, _, _, n_sys, _, _, _ = sys_results[domain]["coverage_score"]
        rp_r, rs_r, _, _, _,     _, _, _ = sys_results[domain]["estimated_coverage_ratio"]
        sys_rows.append([
            domain,
            _fmt(rp_s), _fmt(rs_s),
            _fmt(rp_r), _fmt(rs_r),
            str(n_sys),
        ])
    # Pooled row — show p-values here only
    rp_s_p, rs_s_p, p_rp_s_p, p_rs_s_p, n_sys_p = pooled_sys_summary["coverage_score"]
    rp_r_p, rs_r_p, p_rp_r_p, p_rs_r_p, _        = pooled_sys_summary["estimated_coverage_ratio"]
    sys_rows.append([
        "ALL (pooled)",
        _fmt_rp(rp_s_p, p_rp_s_p), _fmt_rp(rs_s_p, p_rs_s_p),
        _fmt_rp(rp_r_p, p_rp_r_p), _fmt_rp(rs_r_p, p_rs_r_p),
        str(n_sys_p),
    ])
    print(render_table(sys_headers, sys_rows))

    # ------------------------------------------------------------------ #
    # Print: Per-system detail (system means per domain)                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("Per-system means (score / ratio) by domain")
    print("=" * 70)
    for domain in DOMAINS:
        if domain not in sys_results:
            continue
        _, _, _, _, _, ma_s, mb_s, labels_s = sys_results[domain]["coverage_score"]
        _, _, _, _, _, ma_r, mb_r, _        = sys_results[domain]["estimated_coverage_ratio"]
        if not labels_s:
            continue
        print(f"\n  [{domain}]")
        detail_headers = ["system", "score_A", "score_B", "ratio_A", "ratio_B"]
        detail_rows = []
        # Build ratio lookup keyed by label for safe alignment
        ratio_by_label = dict(zip(
            sys_results[domain]["estimated_coverage_ratio"][7],
            zip(ma_r, mb_r),
        ))
        for i, sys_name in enumerate(labels_s):
            short = sys_name.split("/", 1)[-1]
            if len(short) > 50:
                short = short[:47] + "..."
            ra, rb = ratio_by_label.get(sys_name, (None, None))
            detail_rows.append([
                short,
                _fmt(ma_s[i]),
                _fmt(mb_s[i]),
                _fmt(ra),
                _fmt(rb),
            ])
        print(render_table(detail_headers, detail_rows))

    print()


if __name__ == "__main__":
    main()
