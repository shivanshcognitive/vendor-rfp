"""
demo_validation_error_case.py
------------------------------
Standalone demonstration of tools/validation_tool.py's error-handling
path, satisfying the project brief's requirement for "at least one
validation/error case" demonstration -- independent of any LLM call, so
it needs no API key and always produces the same output.

This does NOT simulate an LLM. It constructs a deliberately malformed JSON
string -- three kinds of mistake a real LLM can plausibly make -- and
shows validate_and_normalize() catching all three before they ever reach
the Ranking Tool:

  1. An out-of-range score (15 out of a max of 10).
  2. A missing criterion (the LLM simply didn't return one).
  3. Fabricated evidence -- a claimed quote that does not actually appear
     anywhere in the supplier's proposal text (the evidence-groundedness
     check; see is_evidence_grounded() in tools/validation_tool.py).

This is a direct stress-test of the safety net, which is arguably a
clearer demonstration than waiting for a live model to happen to make a
mistake.

Run: python demo_validation_error_case.py
"""

import json
from database.db_setup import init_db, get_active_criteria
from tools import validation_tool

REAL_PROPOSAL_EXCERPT = """
Apex Systems proposes a modular, cloud-native platform architecture
designed for high scalability and long-term maintainability. Our
architecture uses a microservices design with independent scaling of the
ingestion, scoring, and reporting layers. Delivery is organized into four
phases across 16 weeks: Discovery, Core Build, Integration & Hardening,
and Go-Live. Apex maintains SOC 2 Type II and ISO 27001 certifications,
with role-based access control and quarterly penetration testing.
"""


def main():
    init_db()
    active_criteria = get_active_criteria()

    print("=== Active criteria (source of truth for validation) ===")
    for c in active_criteria:
        print(f"  id={c['criterion_id']}  {c['name']!r}  weight={c['weight']}%  max_score={c['max_score']}")
    print()

    # A deliberately malformed "raw LLM response" -- three kinds of mistake
    # a real model can plausibly make:
    #   1. "Security & Compliance" (criterion_id=4) scored 15, above its
    #      max_score of 10.
    #   2. "Support & Experience" (criterion_id=5) is missing entirely.
    #   3. "Commercial Value" (criterion_id=3)'s evidence is fabricated --
    #      it claims a specific pricing detail that never appears in the
    #      real proposal excerpt above.
    malformed_raw_response = json.dumps({
        "supplier_name": "Demo Supplier",
        "criteria": [
            {"criterion_id": 1, "score": 8, "max_score": 10,
             "justification": "Strong architecture description.",
             "evidence": "Microservices design with independent scaling of the ingestion, scoring, and reporting layers."},
            {"criterion_id": 2, "score": 7, "max_score": 10,
             "justification": "Clear phased plan.",
             "evidence": "Four-phase delivery schedule across 16 weeks."},
            {"criterion_id": 3, "score": 6, "max_score": 10,
             "justification": "Pricing itemized but high.",
             "evidence": "Total year-1 cost of $250,000 with a 10% early-payment discount."},  # FABRICATED
            {"criterion_id": 4, "score": 15, "max_score": 10,  # OUT OF RANGE
             "justification": "Certifications look strong.",
             "evidence": "SOC 2 Type II and ISO 27001 certifications mentioned."},
            # criterion_id 5 is MISSING entirely
        ],
        "risks": ["Schedule risk if credentials are delayed"],
        "overall_summary": "A strong proposal overall.",
    })

    print("=== Deliberately malformed input (simulating three real LLM mistakes) ===")
    print(malformed_raw_response)
    print()

    normalized, warnings = validation_tool.validate_and_normalize(
        malformed_raw_response, active_criteria, "Demo Supplier",
        proposal_text=REAL_PROPOSAL_EXCERPT,
    )

    print("=== Validation Tool warnings raised ===")
    if not warnings:
        print("  (none -- this should not happen for this deliberately broken input)")
    for w in warnings:
        print(f"  - {w}")
    print()

    print("=== Normalized output the Ranking Tool actually receives ===")
    for c in normalized:
        print(f"  {c['name']:25s} score={c['score']:>5}/{c['max_score']}  "
              f"(clipped/defaulted/flagged safely, never exceeds max_score)")
    print()

    assert len(warnings) == 3, f"Expected exactly 3 warnings, got {len(warnings)}: {warnings}"
    assert any("exceeded max_score" in w for w in warnings), "Expected an out-of-range warning"
    assert any("missing from LLM output" in w for w in warnings), "Expected a missing-criterion warning"
    assert any("not appear to be grounded" in w for w in warnings), "Expected an evidence-groundedness warning"
    assert all(c["score"] <= c["max_score"] for c in normalized), "A score still exceeds max_score!"

    print("CONFIRMED: all three deliberate errors were caught and safely handled --")
    print("the out-of-range score was clipped to max_score, the missing criterion")
    print("was defaulted to 0 and flagged, and the fabricated evidence was flagged")
    print("as not grounded in the supplier's actual proposal text. None of these")
    print("bad values reach tools/ranking_tool.py unmodified/unflagged.")


if __name__ == "__main__":
    main()
