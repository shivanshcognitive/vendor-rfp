"""
ranking_tool.py
Ranking Tool: pure deterministic Python. No LLM calls happen here.
Implements exactly the formulas and tie-break order from the project brief.
"""

from datetime import datetime


def compute_absolute_score(normalized_criteria: list) -> float:
    """Sum of (criterion score / max_score) * weight, across all criteria."""
    total = 0.0
    for c in normalized_criteria:
        if c["max_score"] > 0:
            total += (c["score"] / c["max_score"]) * c["weight"]
    return round(total, 4)


def compute_benchmarks(all_suppliers_criteria: dict) -> dict:
    """
    all_suppliers_criteria: {supplier_name: [normalized_criteria...]}
    Returns {criterion_id: benchmark_score} = highest valid score observed
    for that criterion across all suppliers.
    """
    benchmarks = {}
    for supplier, criteria in all_suppliers_criteria.items():
        for c in criteria:
            cid = c["criterion_id"]
            benchmarks[cid] = max(benchmarks.get(cid, 0.0), c["score"])
    return benchmarks


def compute_gaps_and_relative(normalized_criteria: list, benchmarks: dict) -> list:
    """Adds 'benchmark', 'gap', and 'relative_pct' to each criterion dict."""
    enriched = []
    for c in normalized_criteria:
        cid = c["criterion_id"]
        benchmark = benchmarks.get(cid, 0.0)
        gap = round(c["score"] - benchmark, 2)

        if benchmark == 0:
            # safe handling: no valid peer signal for this criterion
            relative_pct = 100.0 if c["score"] == 0 else 0.0
        else:
            relative_pct = round((c["score"] / benchmark) * 100, 2)

        enriched.append({**c, "benchmark": benchmark, "gap": gap, "relative_pct": relative_pct})
    return enriched


def compute_ppi(enriched_criteria: list) -> float:
    """Weighted average of criterion relative-performance percentages."""
    total_weight = sum(c["weight"] for c in enriched_criteria)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(c["relative_pct"] * c["weight"] for c in enriched_criteria)
    return round(weighted_sum / total_weight, 2)


def _parse_date(d):
    if isinstance(d, str):
        return datetime.fromisoformat(d)
    return d


def rank_suppliers(supplier_records: list) -> list:
    """
    supplier_records: list of dicts, each with:
        supplier_name, submission_date (ISO str), experience_rating,
        ppi, absolute_score, criteria (enriched list)

    Applies the mandatory tie-break order:
      1) Higher PPI first
      2) Earlier submission date
      3) Higher historical experience rating
      4) Supplier name ascending
    Then assigns sequential final_rank 1, 2, 3...
    """
    def sort_key(r):
        return (
            -r["ppi"],                          # higher PPI first
            _parse_date(r["submission_date"]),   # earlier date first
            -r["experience_rating"],             # higher experience first
            r["supplier_name"].lower(),          # name ascending
        )

    ranked = sorted(supplier_records, key=sort_key)
    for i, r in enumerate(ranked, start=1):
        r["final_rank"] = i
    return ranked
