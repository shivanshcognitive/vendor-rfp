"""
validation_tool.py
Validation Tool: parses the raw LLM JSON string, checks it against a strict
schema, fills in missing criteria, clips out-of-range scores, checks that
claimed evidence is actually grounded in the source proposal text, and
records warnings for all of the above. This is the ONLY place malformed
or ungrounded LLM output gets fixed/flagged -- once data passes through
here, ranking_tool.py can trust it completely.
"""

import json
import re
from typing import List, Optional
from pydantic import BaseModel, ValidationError, field_validator

# Common short words excluded from the evidence-groundedness word-overlap
# check below, so grounding isn't judged on articles/prepositions that
# would trivially "match" almost any proposal text.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "this", "that", "by", "as", "at", "from", "its", "it",
    "was", "were", "be", "been", "has", "have", "had", "will", "their",
}


class CriterionResult(BaseModel):
    criterion_id: int
    score: float
    max_score: float
    justification: str = ""
    evidence: str = ""

    @field_validator("score")
    @classmethod
    def score_non_negative(cls, v):
        return max(0.0, v)


class LLMScorecard(BaseModel):
    supplier_name: str
    criteria: List[CriterionResult]
    risks: Optional[List[str]] = []
    overall_summary: Optional[str] = ""


def _normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def is_evidence_grounded(evidence: str, proposal_text: str, min_word_overlap: float = 0.5) -> bool:
    """
    Cheap, RAG-adjacent (but not RAG) groundedness check: does the LLM's
    claimed 'evidence' string actually appear in the source proposal text,
    or at least substantially overlap with it? This is deliberately not a
    retrieval system -- there's nothing to retrieve, the whole proposal is
    already in the prompt -- it's a post-hoc sanity check that the model
    didn't fabricate a quote.

    Returns True (grounded) if either:
      - the normalized evidence string is a literal substring of the
        normalized proposal text, or
      - at least `min_word_overlap` fraction of the evidence's
        significant words (length > 3, not a stopword) appear anywhere
        in the proposal text.

    Returns True (i.e. does not flag) when there's nothing meaningful to
    check -- empty evidence/proposal, or an evidence string with no
    significant words -- since there's no basis to call it "ungrounded"
    in that case.
    """
    if not evidence or not proposal_text:
        return True

    norm_evidence = _normalize_text(evidence)
    norm_proposal = _normalize_text(proposal_text)

    if norm_evidence in norm_proposal:
        return True

    words = [w for w in re.findall(r"[a-z0-9]+", norm_evidence)
             if len(w) > 3 and w not in _STOPWORDS]
    if not words:
        return True

    present = sum(1 for w in words if w in norm_proposal)
    return (present / len(words)) >= min_word_overlap


def validate_and_normalize(raw_json: str, active_criteria: list, supplier_name: str,
                            proposal_text: str = None):
    """
    Returns (normalized_criteria: list[dict], warnings: list[str])

    normalized_criteria has exactly one entry per active criterion, with:
      criterion_id, name, weight, max_score, score (clipped 0..max_score),
      justification, evidence

    proposal_text: optional. When given, each criterion's claimed evidence
    is checked against it with is_evidence_grounded() and a warning is
    recorded (not a score change -- there's no sensible "clip" for
    fabricated text) if it doesn't appear to be grounded. Omit this
    argument to skip the groundedness check entirely (e.g. if the caller
    doesn't have the proposal text on hand for some reason).
    """
    warnings = []
    parsed_map = {}  # criterion_id -> CriterionResult

    # --- Step 1: parse JSON safely ---
    try:
        # Strip accidental markdown fences some LLMs add
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as e:
        warnings.append(f"Could not parse LLM JSON output ({e}). "
                         f"All criteria defaulted to 0.")
        data = {"supplier_name": supplier_name, "criteria": [], "risks": [], "overall_summary": ""}

    # --- Step 2: schema validation ---
    try:
        scorecard = LLMScorecard(**data)
        for cr in scorecard.criteria:
            parsed_map[cr.criterion_id] = cr
    except ValidationError as e:
        warnings.append(f"Schema validation issues: {e.error_count()} field error(s) "
                         f"detected; affected entries were skipped.")
        # Best-effort: pull whatever criteria entries do individually validate
        for item in data.get("criteria", []):
            try:
                cr = CriterionResult(**item)
                parsed_map[cr.criterion_id] = cr
            except (ValidationError, TypeError):
                continue

    # --- Step 3: reconcile against the active criteria list (source of truth) ---
    normalized = []
    for c in active_criteria:
        cid = c["criterion_id"]
        max_score = c["max_score"]

        if cid in parsed_map:
            cr = parsed_map[cid]
            score = cr.score
            justification = cr.justification
            evidence = cr.evidence

            # clip out-of-range scores
            if score > max_score:
                warnings.append(
                    f'"{c["name"]}": LLM score {score} exceeded max_score '
                    f"{max_score}; clipped to {max_score}."
                )
                score = max_score
            if cr.max_score != max_score:
                warnings.append(
                    f'"{c["name"]}": LLM reported max_score {cr.max_score}, '
                    f"expected {max_score}; using configured max_score."
                )

            # evidence-groundedness check (only if proposal_text was given)
            if proposal_text is not None and evidence and not is_evidence_grounded(evidence, proposal_text):
                warnings.append(
                    f'"{c["name"]}": claimed evidence does not appear to be '
                    f"grounded in the supplier's proposal text; flagged for "
                    f"manual review."
                )
        else:
            warnings.append(
                f'"{c["name"]}": missing from LLM output; defaulted to 0 '
                f"and flagged for manual review."
            )
            score = 0.0
            justification = "MISSING - not returned by LLM"
            evidence = ""

        normalized.append({
            "criterion_id": cid,
            "name": c["name"],
            "weight": c["weight"],
            "max_score": max_score,
            "score": round(score, 2),
            "justification": justification,
            "evidence": evidence,
        })

    return normalized, warnings
