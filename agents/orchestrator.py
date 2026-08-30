"""
orchestrator.py
Orchestrator Agent (Direct engine): controls the workflow end-to-end,
calling each tool in the required order with plain Python function calls
-- no agent framework. This module has no UI code -- Streamlit (app.py)
only calls run_batch_evaluation() and renders whatever it returns.

Makes a real LLM call for every supplier -- there is no mock/offline mode
anywhere in this pipeline. If no API key resolves (see
tools.llm_tool.resolve_provider()), run_batch_evaluation() raises a
ValueError immediately, before creating a run row or touching the database.

Robustness techniques applied here (see tools/llm_tool.py and
tools/validation_tool.py for the implementations): structured-output
enforcement on the LLM call, evidence-groundedness checking on the
Validation Tool, and retry/backoff on transient LLM errors. All three stay
within "LLM judges content, Python does everything else" -- none change
what the LLM is allowed to decide.

Pipeline (matches the brief's 10-step architecture):
  1. Setup      -> criteria already loaded by caller
  2. Input      -> caller supplies supplier_inputs (name, date, experience, pdf bytes)
  3. Batch      -> create rfp_run row / run id
  4. Evaluate   -> extract text, build prompt, call LLM   (per supplier)
  5. Validate   -> parse + normalize LLM JSON              (per supplier)
  6. Score      -> absolute weighted score                 (per supplier)
  7. Benchmark  -> best score per criterion across suppliers
  8. Rank       -> PPI, tie-breaks, sequential rank
  9. Persist    -> write to SQLite under one rfp_run_id
  10. Present   -> caller (Streamlit) renders the returned structure
"""

import uuid
import json
from datetime import datetime, timezone

from tools import pdf_tool, llm_tool, validation_tool, ranking_tool
from database.db_setup import get_connection


def create_run() -> str:
    """Step 3: Batch. Creates a new rfp_runs row and returns its id."""
    run_id = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        "INSERT INTO rfp_runs (rfp_run_id, created_at, status) VALUES (?, ?, ?)",
        (run_id, datetime.now(timezone.utc).isoformat(), "in_progress"),
    )
    conn.commit()
    conn.close()
    return run_id


def run_batch_evaluation(supplier_inputs: list, active_criteria: list,
                          model: str = None, api_key: str = None, base_url: str = None,
                          max_tokens: int = None, provider: str = None,
                          progress_callback=None) -> dict:
    """
    supplier_inputs: list of dicts:
        {
          "supplier_name": str,
          "submission_date": "YYYY-MM-DD",
          "experience_rating": float (e.g. 0-10),
          "pdf_bytes": bytes  (raw uploaded PDF content)
        }
    active_criteria: list of dicts from database (criterion_id, name, weight, max_score)
    model/api_key/base_url/max_tokens: passed straight to
        tools.llm_tool.resolve_provider()/call_llm_live() (OpenRouter-first,
        then Anthropic, then OpenAI; see tools/llm_tool.py for the full
        resolution order). max_tokens caps the LLM's OUTPUT length per
        supplier call (defaults to 800 if not given -- lower it, e.g. 500,
        if you're on a tight token/credit budget). Leave all four None to
        resolve purely from environment variables.
    progress_callback: optional fn(step:int, total:int, message:str) for UI progress bars

    Makes a real LLM call for every supplier -- there is no offline mode.
    Raises ValueError immediately (before creating a run row) if no API key
    resolves via tools.llm_tool.resolve_provider().

    Returns a dict:
        {
          "rfp_run_id": str,
          "created_at": iso str,
          "results": [ per-supplier enriched record, sorted/ranked ],
          "criteria_used": active_criteria,
        }
    """
    total_steps = len(supplier_inputs) + 3  # evaluate-each + benchmark + rank + persist
    step = 0

    def tick(msg):
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(step, total_steps, msg)

    # Fail fast, before creating a run row or touching the DB, if no LLM
    # provider can be resolved -- avoids a half-created run on a config error.
    llm_tool.resolve_provider(model=model, api_key=api_key, base_url=base_url, provider=provider)

    # Step 3: Batch
    run_id = create_run()

    # Step 4 + 5 + 6: Evaluate, Validate, Score -- per supplier
    all_suppliers_criteria = {}
    per_supplier_meta = {}
    per_supplier_warnings = {}

    for s in supplier_inputs:
        name = s["supplier_name"]
        tick(f"Extracting & evaluating: {name}")

        proposal_text = pdf_tool.extract_text_from_pdf(s["pdf_bytes"])
        raw_json = llm_tool.evaluate_supplier(
            name, proposal_text, active_criteria,
            model=model, api_key=api_key, base_url=base_url, max_tokens=max_tokens,
            provider=provider,
        )
        # proposal_text is passed through so the Validation Tool can run
        # its evidence-groundedness check (see tools/validation_tool.py) --
        # a cheap, RAG-adjacent-but-not-RAG technique that flags evidence
        # the LLM claims but that doesn't actually appear in the document.
        normalized, warnings = validation_tool.validate_and_normalize(
            raw_json, active_criteria, name, proposal_text=proposal_text,
        )

        all_suppliers_criteria[name] = normalized
        per_supplier_warnings[name] = warnings
        per_supplier_meta[name] = {
            "submission_date": s["submission_date"],
            "experience_rating": float(s["experience_rating"]),
            "raw_llm_response": raw_json,
        }

    # Step 7: Benchmark (needs all suppliers' scores first)
    tick("Calculating peer benchmarks")
    benchmarks = ranking_tool.compute_benchmarks(all_suppliers_criteria)

    # Step 8: Rank (compute per-supplier enriched criteria, PPI, absolute score, then sort)
    tick("Scoring, computing PPI, applying tie-breaks")
    supplier_records = []
    for name, normalized in all_suppliers_criteria.items():
        enriched = ranking_tool.compute_gaps_and_relative(normalized, benchmarks)
        absolute_score = ranking_tool.compute_absolute_score(normalized)
        ppi = ranking_tool.compute_ppi(enriched)

        supplier_records.append({
            "supplier_name": name,
            "submission_date": per_supplier_meta[name]["submission_date"],
            "experience_rating": per_supplier_meta[name]["experience_rating"],
            "absolute_score": absolute_score,
            "ppi": ppi,
            "criteria": enriched,
            "warnings": per_supplier_warnings[name],
            "raw_llm_response": per_supplier_meta[name]["raw_llm_response"],
        })

    ranked = ranking_tool.rank_suppliers(supplier_records)

    # Step 9: Persist
    tick("Persisting results to SQLite")
    conn = get_connection()
    for r in ranked:
        result_json = json.dumps(r)
        conn.execute(
            """INSERT INTO supplier_results
               (rfp_run_id, supplier_name, submission_date, experience_rating,
                absolute_score, ppi, final_rank, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, r["supplier_name"], r["submission_date"], r["experience_rating"],
             r["absolute_score"], r["ppi"], r["final_rank"], result_json),
        )
    conn.execute("UPDATE rfp_runs SET status = ? WHERE rfp_run_id = ?", ("completed", run_id))
    conn.commit()
    conn.close()

    return {
        "rfp_run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": ranked,
        "criteria_used": active_criteria,
        "engine": "direct",
    }


def load_run_from_db(run_id: str) -> dict:
    """Step 10 helper: reload a persisted run for display / re-download."""
    conn = get_connection()
    run_row = conn.execute(
        "SELECT * FROM rfp_runs WHERE rfp_run_id = ?", (run_id,)
    ).fetchone()
    if not run_row:
        conn.close()
        return None

    rows = conn.execute(
        "SELECT * FROM supplier_results WHERE rfp_run_id = ? ORDER BY final_rank",
        (run_id,),
    ).fetchall()
    conn.close()

    results = [json.loads(r["result_json"]) for r in rows]
    return {
        "rfp_run_id": run_id,
        "created_at": run_row["created_at"],
        "status": run_row["status"],
        "results": results,
    }


def list_runs() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT rfp_run_id, created_at, status FROM rfp_runs ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
