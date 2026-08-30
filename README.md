# VendorScope — Agentic RFP Evaluation & Supplier Ranking

A classroom mini project: an AI-assisted app that reads supplier RFP PDF
responses, scores them against configurable criteria using a real LLM,
then applies **deterministic Python** for all arithmetic, peer
benchmarking, tie-breaks, and final ranking.

> **Key design rule:** the LLM only judges proposal *content* (scores,
> justification, evidence). It never computes weighted totals, benchmarks,
> gaps, PPI, tie-breaks, or rank — that is 100% deterministic Python, so
> the same validated inputs always produce the same leaderboard.

There is **no mock/offline mode anywhere in this project** — every
evaluation makes a real LLM call. This is deliberate: it keeps the
implementation simple to follow (one code path, not two), and it means a
reviewer never has to wonder whether a given run used simulated data.

The pipeline ships with **two engines**, selectable in the Streamlit
sidebar, both calling the exact same `tools/` modules and both requiring
an API key:

| Engine | How it runs | Where |
|---|---|---|
| **Direct** | Plain Python function calls, no framework | `agents/orchestrator.py` |
| **LangGraph** | An explicit typed-state graph: `start_batch` → `evaluate` (loops once per supplier — the only node that calls an LLM) → `benchmark_and_rank` → `persist` | `agents_langgraph/langgraph_pipeline.py` |

Both engines call `tools/llm_tool.resolve_provider()` before doing
anything else; if no API key resolves, they raise a clear `ValueError`
immediately rather than silently falling back to anything.

---

## 1. Architecture

```
Streamlit UI (app.py)
        │
        ▼
Orchestrator (agents/orchestrator.py, plain Python)
   or a compiled LangGraph (agents_langgraph/langgraph_pipeline.py)
        │
        ├── 1. Setup     → database/db_setup.py        (load active criteria)
        ├── 2. Input     → Streamlit file_uploader + metadata form
        ├── 3. Batch     → create rfp_runs row, new RFP_RUN_ID
        ├── 4. Evaluate  → tools/pdf_tool.py  (extract text)
        │                → tools/llm_tool.py (real LLM call → JSON scorecard)
        ├── 5. Validate  → tools/validation_tool.py (schema, clipping,
        │                  evidence-groundedness check, warnings)
        ├── 6. Score     → tools/ranking_tool.py (absolute weighted score)
        ├── 7. Benchmark → tools/ranking_tool.py (best score per criterion)
        ├── 8. Rank      → tools/ranking_tool.py (PPI, tie-breaks, sequential rank)
        ├── 9. Persist   → database/db_setup.py (supplier_results table)
        └── 10. Present  → Streamlit tabs (Leaderboard, Scorecard, Run Details)
```

### Why LangGraph, not AutoGen/CrewAI/plain LangChain

This project previously used AutoGen for its second engine; this version
replaces it with LangGraph. The reasoning, so it's explicit rather than
assumed:

- **LangGraph fits the brief's own diagram almost literally.** The
  10-step architecture above is already a directed graph with exactly one
  loop (evaluate-then-validate, once per supplier). LangGraph's
  `StateGraph` models that shape directly — typed state passed node to
  node, one conditional edge for the loop — rather than requiring an
  agent-conversation metaphor to explain what's really just sequential
  data processing.
- **CrewAI was not used.** Its core value proposition is multiple agents
  with roles/goals delegating to each other autonomously. This brief has
  exactly one point of LLM judgment and explicitly states everything else
  "must not decide" anything — forcing CrewAI on here would mean inventing
  agent personas for tasks (validation, ranking) that are supposed to have
  zero discretion, which works against the brief's own constraint rather
  than demonstrating it.
- **Plain LangChain (without LangGraph) was not used.** Its main offer for
  this project would be structured-output parsing and PDF loading —
  `tools/validation_tool.py` and `tools/pdf_tool.py` already do both by
  hand, in a form a reviewer can read line by line, rather than trusting
  an `OutputParser`'s internals.
- **RAG was not used.** Supplier proposals are 2–4 pages and fit entirely
  in one prompt; there's no external corpus to retrieve against. RAG
  solves "the model can't see enough relevant context," which isn't the
  problem here. It would become relevant if proposals scaled past ~20
  pages, or if a real vendor-performance database were introduced to
  cross-reference against.

### Two engines, why keep both

`agents/orchestrator.py` (Direct) is the simplest possible reading of the
brief — plain function calls, nothing to learn to follow it.
`agents_langgraph/langgraph_pipeline.py` demonstrates the same pipeline
through an explicit graph/state-machine framework, which the brief's own
"suggested tools" list names as an acceptable alternative to plain Python.
Both call the identical `tools/` modules, so neither engine's code path
ever re-implements scoring, benchmarking, or ranking independently.

---

## 2. Folder structure

```
vendorscope/
├── app.py                          # Streamlit entry point (all 5 screens)
├── demo_validation_error_case.py   # Standalone Validation Tool error-case demo
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml                 # Theme colors/fonts (clean-enterprise light palette)
│   └── secrets.toml.example        # copy → secrets.toml for your API key
├── assets/
│   ├── generate_rubik_icon.py      # Generates the cube logo/favicon (Pillow)
│   └── rubik_icon.png              # The generated cube logo/favicon
├── database/
│   ├── db_setup.py                 # schema + seed script (run standalone or auto-run by app.py)
│   └── rfp_evaluation.db           # created at runtime, not committed
├── tools/
│   ├── pdf_tool.py                 # Document Tool
│   ├── llm_tool.py                 # Evaluation Agent (structured output, retry/backoff)
│   ├── validation_tool.py          # Validation Tool (schema, clipping, groundedness check)
│   └── ranking_tool.py             # Ranking Tool
├── agents/
│   └── orchestrator.py             # Direct engine
├── agents_langgraph/
│   └── langgraph_pipeline.py       # LangGraph engine
├── notebooks/
│   └── VendorScope_Colab.ipynb     # Self-contained Colab runner (both engines + demos)
├── sample_data/
│   ├── generate_supplier_pdfs.py   # regenerates the 4 synthetic PDFs below
│   └── supplier_pdfs/
│       ├── Apex_Systems.pdf
│       ├── BrightPath_Tech.pdf
│       ├── NexaWorks.pdf
│       └── Orbit_Digital.pdf
├── sample_output/
│   └── README.md                   # how to generate sample_run_result.json with your key
└── screenshots/
    └── README.md                   # which 5 screens to capture, see §8 below
```

---

## 3. Setup

### Get an API key (required — there is no offline mode)
Pick **one** of these, checked in this order by
`tools/llm_tool.resolve_provider()`:
```bash
export OPENROUTER_API_KEY="sk-or-..."       # recommended: openrouter.ai/keys
# or
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```
OpenRouter is recommended: one key works across many providers/models. **One
important nuance:** the default model this project uses
(`openai/gpt-4o-mini`) is a *paid* model on OpenRouter, billed against your
OpenRouter credit balance — it is not one of OpenRouter's free-tier
options. If you have a $0 balance, calls to it fail with a permanent
`APIStatusError` (commonly a 402 "insufficient credits"), and no amount of
retrying fixes that. To actually run this at zero cost, either add a small
credit balance to your OpenRouter account, or override the model to a
genuine free one (any current `:free`-suffixed model at
[openrouter.ai/models](https://openrouter.ai/models), e.g.
`meta-llama/llama-3.3-70b-instruct:free`) via the "Model override" field in
the app's Advanced settings, or by passing `model=...` directly to either
engine.

### Local run
```bash
git clone <your-repo-url>
cd vendorscope
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

export OPENROUTER_API_KEY="sk-or-..."   # see above

python sample_data/generate_supplier_pdfs.py   # optional: regenerate the 4 synthetic PDFs
python database/db_setup.py                     # optional: app.py also does this automatically

streamlit run app.py
```
The sidebar's provider dropdown (OpenRouter / Anthropic / OpenAI) also
accepts a key pasted directly into the app for a one-off session. **This
is safe on a public deployment**: a key typed into the sidebar is stored
only in that visitor's own `st.session_state`, never written to the
shared server environment, so it can't leak to or be used by other
concurrent visitors of the same app instance.

### Deploy to Streamlit Community Cloud
1. Push this repo to GitHub.
2. On share.streamlit.io, create a new app pointing at `app.py`.
3. Add `OPENROUTER_API_KEY` (or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`)
   under the app's Secrets settings — required, since there's no offline
   mode for visitors to fall back to.
4. Deploy — `requirements.txt` and `database/db_setup.py` (auto-run by
   `app.py`) handle the rest.

**Security note on API keys and public deployments:** on Streamlit
Community Cloud, one running app instance is shared by every concurrent
visitor's browser session. `app.py` mirrors `st.secrets` into `os.environ`
once at startup — that's intentional, since it represents the app owner's
own key, deliberately shared with all visitors. A visitor's own key,
typed into the sidebar, is never written to `os.environ` — it's kept in
that visitor's own `st.session_state` and passed explicitly as
`api_key=...` into the evaluation call for their requests only.

### Using Google Colab
Streamlit doesn't run natively inside a Colab notebook cell, so
`notebooks/VendorScope_Colab.ipynb` is for running/inspecting the pipeline
logic — not the UI. It's fully self-contained (every project file is
recreated via `%%writefile` cells, no zip or GitHub clone needed):
- Steps 1–5 install dependencies, write the project files, seed the
  database, and generate the synthetic PDFs.
- Step 6 runs the standalone validation/error-case demo — no API key
  needed.
- Step 7 onward requires your API key and runs both engines for real,
  printing actual model-generated scores, justifications, and evidence.

For the actual submission, deploy the UI via GitHub + Streamlit Community
Cloud as above.

---

## 4. Database design

| Table | Fields |
|---|---|
| `evaluation_criteria` | `criterion_id, name, description, weight, max_score, is_active` |
| `rfp_runs` | `rfp_run_id (UUID), created_at, status` |
| `supplier_results` | `rfp_run_id, supplier_name, submission_date, experience_rating, absolute_score, ppi, final_rank, result_json` |

Default seeded criteria (weights sum to 100%):

| Criterion | Weight |
|---|---|
| Technical Capability | 30% |
| Implementation Plan | 20% |
| Commercial Value | 20% |
| Security & Compliance | 20% |
| Support & Experience | 10% |

Criteria can be activated/deactivated or reweighted directly in the
`evaluation_criteria` table without touching any prompt or app code — the
app always reads whatever is currently active and validates that active
weights total 100% before allowing an evaluation.

---

## 5. Formulas (all computed in `tools/ranking_tool.py`, pure Python)

- **Absolute weighted score** = Σ `(criterion_score / max_score) × weight`
- **Criterion benchmark** = highest valid score observed for that criterion across all suppliers in the batch
- **Criterion gap** = `supplier_score − benchmark` (0 for the leader, negative otherwise)
- **Relative performance %** = `(supplier_score / benchmark) × 100`
  — *safe handling*: if `benchmark == 0`, relative % is defined as `100` when the supplier also scored 0, else `0`
- **Peer Performance Index (PPI)** = weighted average of each criterion's relative-performance % (weights = criterion weights)

### Tie-break order (mandatory, applied before assigning ranks)
1. Higher PPI first
2. Earlier submission date
3. Higher historical experience rating
4. Supplier name, ascending alphabetically

Rank 1, 2, 3... is assigned only after this stable sort — see
`ranking_tool.rank_suppliers()`.

---

## 6. Robustness techniques (stay entirely within "LLM judges content, Python does everything else")

| Technique | What it does | Where |
|---|---|---|
| **Structured output enforcement** | OpenAI-compatible calls (OpenRouter/OpenAI) request `response_format={"type":"json_schema",...}`; Anthropic calls force a tool-use call against the same schema. Guarantees schema-valid JSON at the API level when the provider honors it, cutting malformed-response noise at the source. Falls back to a plain call if the provider rejects the parameter. `tools/validation_tool.py` still re-validates everything regardless — never trust the wire even with a schema. | `tools/llm_tool.py` — `SCORECARD_SCHEMA`, `_call_openai_compatible()`, `_call_anthropic()` |
| **Evidence-groundedness check** | A cheap, RAG-*adjacent but not RAG* technique: after the LLM claims an `evidence` string per criterion, checks whether that text actually appears (or substantially word-overlaps with) the real extracted proposal text. If not, flags a warning — same pattern as the existing out-of-range-score clipping, just checking groundedness instead of numeric range. Directly strengthens the brief's own "evidence-grounded JSON" requirement, at zero extra LLM calls. | `tools/validation_tool.py` — `is_evidence_grounded()` |
| **Retry/backoff on transient failures** | Up to 2 retries (3 attempts total) with exponential backoff (1.5s, 3s) on rate-limit/5xx errors; non-retryable errors (bad key, malformed request) fail immediately. Standard production hygiene, zero architectural change. | `tools/llm_tool.py` — `_with_retries()`, `_is_retryable()` |
| **Self-consistency / majority voting** — *considered and declined* | Calling the LLM 2-3× per supplier and taking the median score. Deliberately **not implemented**: it would triple per-supplier token/credit cost for a marginal robustness gain, which cuts against this project's own token-budget goals. Noted here as a technique evaluated and set aside, not one worth building at this scope. | — |

---

## 7. Assumptions

- "Any JSON-capable LLM" is implemented via `tools/llm_tool.resolve_provider()`,
  which auto-resolves OpenRouter (`openai/gpt-4o-mini` default, recommended
  for one-key/many-model flexibility), direct Anthropic (`claude-sonnet-4-6`),
  or direct OpenAI (`gpt-4o-mini`) from whichever API key is set.
- Experience rating is entered as a 0–10 float by the user at upload time
  (the brief doesn't fix a scale).
- `max_score` per criterion is fixed at 10 in the seed data but is fully
  configurable per row in `evaluation_criteria`.
- Every evaluation makes a real LLM call — there is intentionally no
  offline/mock mode. The required "validation/error case" demonstration is
  instead a standalone stress-test of the Validation Tool
  (`demo_validation_error_case.py`) covering three deliberate error types
  (out-of-range score, missing criterion, fabricated/ungrounded evidence),
  which needs no API key and always produces the same output.
- PDFs are processed entirely in memory (`BytesIO`) from Streamlit's
  uploader — nothing is written to disk during evaluation.
- The evidence-groundedness check uses substring + word-overlap matching,
  not semantic similarity — a paraphrase that preserves the proposal's
  actual key terms will pass; a fabricated claim with unrelated wording
  will be flagged. This is intentionally simple and inspectable rather
  than another LLM call.

---

## 8. Testing / reproducibility

`demo_validation_error_case.py` is the reproducible, zero-cost proof that
the Validation Tool's error-handling path works: it feeds
`tools/validation_tool.py` a deliberately malformed scorecard with three
planted errors (an out-of-range score, a missing criterion, and
fabricated evidence for a real proposal excerpt) and asserts all three get
caught and safely handled before reaching the Ranking Tool. Run it
directly:
```bash
python demo_validation_error_case.py
```

**Successful run + validation/error case (submission requirement):**
`notebooks/VendorScope_Colab.ipynb` covers both explicitly — Step 6 runs
`demo_validation_error_case.py` (no API key needed), and Step 8 runs a
full successful batch independently through each engine (API key
required; each engine's leaderboard stands on its own — the two are not
compared against each other, since two separate live LLM calls aren't
guaranteed to return identical scores). Step 9 exports both engines'
completed runs to `sample_output/`.

Everything *downstream* of the LLM call is deterministic: given the same
validated scorecards, `tools/ranking_tool.py` always produces the same
formulas and ordering. This was verified directly: running the ranking
pipeline 30 times (10 with fixed supplier order, 20 with randomly
shuffled order) on the same validated input produced byte-identical
output every time, and every final score was confirmed to be traceable
back to its own criteria list (weight × score/max_score, and weight ×
relative_pct) within floating-point rounding tolerance.

---

## 9. Screenshots

**Not included in this package** — capturing real screenshots requires a
running instance (local `streamlit run app.py` or the deployed Streamlit
Community Cloud URL) that a human can view in a browser. Once you have the
app running (locally or deployed, with an API key configured), capture
these 5 screens — they map directly to the app's 5 tabs — and drop them
into the `screenshots/` folder at the repo root:

| # | Filename (suggested) | Tab | What to show |
|---|---|---|---|
| 1 | `01_criteria.png` | Criteria | The active-criteria table with weights summing to 100% |
| 2 | `02_supplier_input.png` | Supplier Input | 4 PDFs uploaded, metadata filled in, validation messages visible |
| 3 | `03_leaderboard.png` | Leaderboard | The gold/silver/bronze top-3 podium cards, plus the ranked table + PPI bar chart |
| 4 | `04_scorecard.png` | Detailed Scorecard | One supplier's per-criterion breakdown with evidence/justification expanded |
| 5 | `05_run_details.png` | Run Details | RFP_RUN_ID, tie-break explanation, JSON download button |

Then reference them here, e.g.:
```markdown
![Criteria screen](screenshots/01_criteria.png)
![Leaderboard screen](screenshots/03_leaderboard.png)
```

---

## 10. LangGraph engine details

`agents_langgraph/langgraph_pipeline.py` models the 10-step pipeline as a
compiled `StateGraph`:

```
start_batch → evaluate ─┐
                 ▲       │ (loops until every supplier is done)
                 └───────┘
                         │
                         ▼
              benchmark_and_rank → persist → END
```

- **`evaluate`** is the only node that calls an LLM (`tools/llm_tool.evaluate_supplier()`),
  and only ever judges one supplier's content per visit. A conditional
  edge (`route_after_evaluate`) re-enters this same node once per
  remaining supplier.
- **`benchmark_and_rank`** and **`persist`** only ever call
  `tools/ranking_tool.py` and `database/db_setup.py` — they never see a
  raw LLM response, only the already-validated, normalized criteria the
  `evaluate` node produced.
- Unlike the AutoGen-based engine this project used previously (whose
  `ToolExecutor` ran fixed code through a sandboxed `code_execution_config`
  specifically to demonstrate that validation/ranking logic isn't
  LLM-authored), LangGraph nodes are plain Python functions the graph
  engine calls directly. The same non-authorship guarantee holds by
  construction — there's no separate sandbox-execution step needed to
  prove it.

Install: `pip install langgraph` (already in `requirements.txt`).

Run it directly (requires an API key resolvable by
`tools/llm_tool.resolve_provider()`):
```python
from database.db_setup import init_db, get_active_criteria
from agents_langgraph.langgraph_pipeline import run_langgraph_batch_evaluation

init_db()
result = run_langgraph_batch_evaluation(supplier_inputs, get_active_criteria())
```
Or just pick **"LangGraph"** in the Streamlit sidebar before clicking
"Evaluate Batch" — `app.py` calls whichever engine you select with the
identical inputs/outputs shape.

---

## 11. Troubleshooting: "APIStatusError" / app crashes on Evaluate

If Streamlit Cloud shows a crash with `openai.APIStatusError` and a
redacted message, this is almost always one of two things:

1. **No/insufficient OpenRouter credit balance.** The default model
   (`openai/gpt-4o-mini`) is paid, not free — see the note in §3 above.
   Fix: add a small credit balance to your OpenRouter account, **or** use
   the "Model override" field in the sidebar's Advanced settings to enter
   a genuine free model (e.g. `meta-llama/llama-3.3-70b-instruct:free` —
   check [openrouter.ai/models](https://openrouter.ai/models) for the
   current list, since free-tier model IDs rotate).
2. **An invalid/expired API key.** Re-check the key in Streamlit Secrets
   or the sidebar field.

**To see the actual (unredacted) error:** on Streamlit Cloud, click
"Manage app" (bottom right) → the deployed app's logs show the real
message the UI redacts for you.

**Why retries don't help here, and won't loop forever either:**
`tools/llm_tool._is_retryable()` only retries genuine transient errors —
rate limits (429) and server errors (500/502/503/529). Client errors like
401 (bad key), 402 (payment required), 403, and 404 (bad model name) fail
immediately, on the first attempt, with no wasted retry delay — retrying
a permanent failure can never succeed, so there's no reason to wait for
it.

---

## 12. Rubric traceability

| Rubric area | Where it's addressed |
|---|---|
| Agentic workflow & tool use | `agents/orchestrator.py` and `agents_langgraph/langgraph_pipeline.py` clearly separate orchestration from each tool call |
| PDF extraction & prompting | `tools/pdf_tool.py`, `tools/llm_tool.py` (dynamic criteria injected into prompt, structured-output schema, evidence-grounded JSON) |
| Validation & scoring | `tools/validation_tool.py` (schema, clipping, groundedness check), `tools/ranking_tool.py`, `demo_validation_error_case.py` |
| Peer ranking & tie-breaks | `tools/ranking_tool.py` (`compute_benchmarks`, `rank_suppliers`) |
| SQLite & persistence | `database/db_setup.py`, both engines' persist step |
| Streamlit UI | `app.py` — 5 tabs matching the 5 required screens, plus a leaderboard podium |
| Documentation & testing | This README + `demo_validation_error_case.py` + synthetic PDFs + Colab notebook |
