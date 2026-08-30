"""
langgraph_pipeline.py
-----------------------
Same 10-step pipeline as agents/orchestrator.py, re-implemented as an
explicit LangGraph StateGraph instead of plain Python function calls.

Why LangGraph fits this brief well: the project's own architecture diagram
(Setup -> Input -> Batch -> Evaluate -> Validate -> Score -> Benchmark ->
Rank -> Persist -> Present) is already a directed graph with one loop
(evaluate-then-validate, once per supplier). LangGraph models exactly
that shape -- typed state passed node to node, with an explicit
conditional edge for the per-supplier loop -- so this engine is a fairly
literal implementation of the brief's own diagram, not a generic
"agentic wrapper" bolted on top of it.

Nodes:
  - start_batch          : creates the rfp_runs row, initializes state
  - evaluate              : Document Tool (extract PDF text) + Evaluation
                             Agent (real LLM call) + Validation Tool, for
                             ONE supplier; loops back to itself via a
                             conditional edge until every supplier has been
                             evaluated
  - benchmark_and_rank   : Ranking Tool -- benchmarks, absolute score, PPI,
                             tie-breaks, sequential rank (fully deterministic)
  - persist               : writes the ranked results to SQLite

Unlike the earlier AutoGen engine's ToolExecutor (which ran fixed code
through a sandboxed code_execution_config specifically to demonstrate that
validation/ranking logic isn't LLM-authored), LangGraph nodes are just
plain Python functions we wrote and the graph engine calls directly -- the
same non-authorship guarantee holds by construction, with no separate
sandbox-execution step needed to demonstrate it. The "evaluate" node is
the ONLY node that calls an LLM (tools.llm_tool.evaluate_supplier());
benchmark_and_rank and persist only ever call tools.ranking_tool /
database.db_setup, which never see a raw LLM response, only already-
validated, normalized data.

Requirements: pip install langgraph
"""

import uuid
import json
from datetime import datetime, timezone
from typing import TypedDict, Optional, Callable, Any

from langgraph.graph import StateGraph, END

from tools import pdf_tool, llm_tool, validation_tool, ranking_tool
from database.db_setup import get_connection


class GraphState(TypedDict):
    # ---- set once at invoke time, read-only from every node's perspective ----
    supplier_inputs: list
    active_criteria: list
    model: Optional[str]
    api_key: Optional[str]
    base_url: Optional[str]
    max_tokens: Optional[int]
    provider: Optional[str]
    progress_callback: Optional[Callable[[int, int, str], Any]]

    # ---- working state, updated node to node ----
    run_id: str
    supplier_index: int
    all_suppliers_criteria: dict
    per_supplier_meta: dict
    per_supplier_warnings: dict

    # ---- final output ----
    ranked: list


def _tick(state: GraphState, step: int, total: int, msg: str):
    cb = state.get("progress_callback")
    if cb:
        cb(step, total, msg)


def _total_steps(state: GraphState) -> int:
    return len(state["supplier_inputs"]) * 2 + 3


def node_start_batch(state: GraphState) -> dict:
    """Step 3: Batch. Creates a new rfp_runs row and initializes the
    per-supplier accumulators the evaluate loop will fill in."""
    run_id = str(uuid.uuid4())
    conn = get_connection()
    conn.execute(
        "INSERT INTO rfp_runs (rfp_run_id, created_at, status) VALUES (?, ?, ?)",
        (run_id, datetime.now(timezone.utc).isoformat(), "in_progress"),
    )
    conn.commit()
    conn.close()
    return {
        "run_id": run_id,
        "supplier_index": 0,
        "all_suppliers_criteria": {},
        "per_supplier_meta": {},
        "per_supplier_warnings": {},
    }


def node_evaluate_one_supplier(state: GraphState) -> dict:
    """Steps 4+5: Evaluate (Document Tool + Evaluation Agent -- the ONLY
    real LLM call in this graph) and Validate (Validation Tool, including
    the evidence-groundedness check), for exactly one supplier per visit
    to this node. The conditional edge below re-enters this same node
    until every supplier has been processed."""
    idx = state["supplier_index"]
    s = state["supplier_inputs"][idx]
    name = s["supplier_name"]
    active_criteria = state["active_criteria"]
    total = _total_steps(state)

    _tick(state, idx * 2 + 1, total, f"[LangGraph] Evaluator scoring: {name}")
    proposal_text = pdf_tool.extract_text_from_pdf(s["pdf_bytes"])
    raw_json = llm_tool.evaluate_supplier(
        name, proposal_text, active_criteria,
        model=state.get("model"), api_key=state.get("api_key"),
        base_url=state.get("base_url"), max_tokens=state.get("max_tokens"),
        provider=state.get("provider"),
    )

    _tick(state, idx * 2 + 2, total, f"[LangGraph] Validating: {name}")
    normalized, warnings = validation_tool.validate_and_normalize(
        raw_json, active_criteria, name, proposal_text=proposal_text,
    )

    all_criteria = dict(state["all_suppliers_criteria"])
    all_criteria[name] = normalized
    per_meta = dict(state["per_supplier_meta"])
    per_meta[name] = {
        "submission_date": s["submission_date"],
        "experience_rating": float(s["experience_rating"]),
        "raw_llm_response": raw_json,
    }
    per_warn = dict(state["per_supplier_warnings"])
    per_warn[name] = warnings

    return {
        "supplier_index": idx + 1,
        "all_suppliers_criteria": all_criteria,
        "per_supplier_meta": per_meta,
        "per_supplier_warnings": per_warn,
    }


def route_after_evaluate(state: GraphState) -> str:
    """Conditional edge: loop back to 'evaluate' while suppliers remain,
    otherwise proceed to benchmarking/ranking."""
    if state["supplier_index"] < len(state["supplier_inputs"]):
        return "evaluate"
    return "benchmark_and_rank"


def node_benchmark_and_rank(state: GraphState) -> dict:
    """Steps 6-8: Score, Benchmark, Rank -- entirely deterministic Python
    via tools/ranking_tool.py. Never touches a raw LLM response, only the
    already-validated, normalized criteria produced by the evaluate node."""
    total = _total_steps(state)
    _tick(state, len(state["supplier_inputs"]) * 2 + 1, total,
          "[LangGraph] Calculating peer benchmarks, scoring, ranking")

    all_suppliers_criteria = state["all_suppliers_criteria"]
    per_supplier_meta = state["per_supplier_meta"]
    per_supplier_warnings = state["per_supplier_warnings"]

    benchmarks = ranking_tool.compute_benchmarks(all_suppliers_criteria)

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
    _tick(state, len(state["supplier_inputs"]) * 2 + 2, total,
          "[LangGraph] Benchmarking, scoring, and ranking complete")
    return {"ranked": ranked}


def node_persist(state: GraphState) -> dict:
    """Step 9: Persist. Writes the ranked results to SQLite under the
    run's rfp_run_id."""
    total = _total_steps(state)
    _tick(state, total, total, "[LangGraph] Persisting results to SQLite")

    run_id = state["run_id"]
    ranked = state["ranked"]
    conn = get_connection()
    for r in ranked:
        conn.execute(
            """INSERT INTO supplier_results
               (rfp_run_id, supplier_name, submission_date, experience_rating,
                absolute_score, ppi, final_rank, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, r["supplier_name"], r["submission_date"], r["experience_rating"],
             r["absolute_score"], r["ppi"], r["final_rank"], json.dumps(r)),
        )
    conn.execute("UPDATE rfp_runs SET status = ? WHERE rfp_run_id = ?", ("completed", run_id))
    conn.commit()
    conn.close()
    return {}


def build_graph():
    """Builds and compiles the StateGraph. Exposed separately from
    run_langgraph_batch_evaluation() so the graph structure can be
    inspected/visualized independently of running it."""
    g = StateGraph(GraphState)
    g.add_node("start_batch", node_start_batch)
    g.add_node("evaluate", node_evaluate_one_supplier)
    g.add_node("benchmark_and_rank", node_benchmark_and_rank)
    g.add_node("persist", node_persist)

    g.set_entry_point("start_batch")
    g.add_edge("start_batch", "evaluate")
    g.add_conditional_edges(
        "evaluate", route_after_evaluate,
        {"evaluate": "evaluate", "benchmark_and_rank": "benchmark_and_rank"},
    )
    g.add_edge("benchmark_and_rank", "persist")
    g.add_edge("persist", END)

    return g.compile()


def run_langgraph_batch_evaluation(supplier_inputs: list, active_criteria: list,
                                    model: str = None, api_key: str = None,
                                    base_url: str = None, max_tokens: int = None,
                                    provider: str = None,
                                    progress_callback=None) -> dict:
    """
    Runs the full 10-step pipeline through the LangGraph engine, always
    with a real LLM call for every supplier (see tools/llm_tool.py -- there
    is no offline mode anywhere in this project). If no API key resolves,
    raises a ValueError before creating a run row or touching the database.

    Same signature/return shape as agents.orchestrator.run_batch_evaluation,
    so app.py can call either engine with matching inputs/outputs.
    """
    # Fail fast, before creating a run row or touching the DB, if no live
    # provider can be resolved -- avoids a half-created run on a config error.
    llm_tool.resolve_provider(model=model, api_key=api_key, base_url=base_url, provider=provider)

    graph = build_graph()
    initial_state: GraphState = {
        "supplier_inputs": supplier_inputs,
        "active_criteria": active_criteria,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "provider": provider,
        "progress_callback": progress_callback,
        "run_id": "",
        "supplier_index": 0,
        "all_suppliers_criteria": {},
        "per_supplier_meta": {},
        "per_supplier_warnings": {},
        "ranked": [],
    }
    # Default recursion_limit (25) comfortably covers small batches (each
    # supplier costs one step through the evaluate loop), but bump it for
    # safety on larger batches -- this only bounds graph steps, not cost.
    final_state = graph.invoke(initial_state, config={"recursion_limit": 200})

    return {
        "rfp_run_id": final_state["run_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": final_state["ranked"],
        "criteria_used": active_criteria,
        "engine": "langgraph",
    }
