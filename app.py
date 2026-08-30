"""
app.py
Streamlit UI for VendorScope -- Agentic RFP Evaluation & Supplier Ranking.

Screens (per project brief section 9):
  - Criteria            : active criteria, weights, max score
  - Supplier input      : multi-PDF upload, metadata, validation messages, Evaluate
  - Leaderboard         : rank, supplier, absolute score, PPI, date, experience
  - Detailed scorecard  : per-criterion score, benchmark, gap, relative %, evidence
  - Run details         : RFP_RUN_ID, warnings, tie-break explanation, JSON download
"""

import os
import json
import base64
import html
import streamlit as st
import pandas as pd

from database.db_setup import init_db, get_active_criteria, active_weight_total
from agents.orchestrator import run_batch_evaluation, list_runs, load_run_from_db
from agents_langgraph.langgraph_pipeline import run_langgraph_batch_evaluation

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "rubik_icon.png")


def _load_logo_b64():
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


LOGO_B64 = _load_logo_b64()

st.set_page_config(
    page_title="VendorScope",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🧩",
    layout="wide",
)
init_db()

# ------------------------------------------------------- theme & typography --
# Clean-enterprise token system: light, muted-blue background with a single
# colorful signature element (the cube logo + rank podium) carrying all the
# visual energy, per a deliberately restrained design brief.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --bg: #F7F9FC;
    --surface: #FFFFFF;
    --border: #DCE3ED;
    --text: #1F2A3C;
    --text-muted: #5B6B82;
    --primary: #2C4870;
    --primary-hover: #1F3452;
    --accent-blue: #4C7EAF;
    --success: #3F8F6B;
    --warning: #C98A3D;
    --gold: #D4AF37;
    --silver: #9AA0A8;
    --bronze: #B08D57;
}

.stApp {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text);
}

h1, h2, h3, h4 {
    font-family: 'Sora', sans-serif !important;
    color: var(--text) !important;
}

/* ---- header banner ---- */
.app-header {
    display: flex;
    align-items: center;
    gap: 22px;
    padding-bottom: 18px;
    margin-bottom: 22px;
    border-bottom: 2px solid var(--border);
}
.app-logo { width: 62px; height: auto; flex-shrink: 0; }
.app-title {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    font-size: 2.05rem;
    color: var(--text);
    margin: 0;
    letter-spacing: -0.01em;
}
.app-tagline {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-muted);
    font-size: 0.97rem;
    margin: 4px 0 0 0;
}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    color: var(--text-muted);
    padding: 10px 6px;
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important;
    border-bottom-color: var(--primary) !important;
}

/* ---- buttons ---- */
.stButton > button {
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    border-radius: 8px;
}
.stButton > button[kind="primary"] {
    background-color: var(--primary);
    border-color: var(--primary);
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--primary-hover);
    border-color: var(--primary-hover);
}
.stDownloadButton > button {
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    border-radius: 8px;
}

/* ---- metrics ---- */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--primary);
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-muted);
}

/* ---- monospace for run IDs / codes ---- */
code {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ---- podium (leaderboard signature element) ---- */
.podium-row {
    display: flex;
    gap: 16px;
    margin: 6px 0 26px 0;
    flex-wrap: wrap;
}
.podium-card {
    flex: 1;
    min-width: 200px;
    border-radius: 12px;
    border-top: 5px solid var(--border);
    background: var(--surface);
    padding: 18px 18px 16px 18px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}
.podium-medal { font-size: 1.7rem; line-height: 1; }
.podium-rank {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 6px;
}
.podium-name {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.12rem;
    color: var(--text);
    margin-top: 2px;
    overflow-wrap: break-word;
}
.podium-ppi {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    color: var(--primary);
    margin-top: 8px;
    font-size: 0.98rem;
}
</style>
""", unsafe_allow_html=True)

# Mirror any Streamlit secrets into the environment so tools/llm_tool.py's
# resolve_provider() can find them (it checks OPENROUTER_API_KEY, then
# ANTHROPIC_API_KEY, then OPENAI_API_KEY -- see that module for details).
try:
    for _key in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if _key in st.secrets:
            # Always re-sync from the CURRENT secret value on every rerun --
            # never skip this because os.environ already has a value. On
            # Streamlit Cloud, the same process reruns the script many times
            # without a full restart; os.environ persists across reruns, so
            # a prior guard here (only set if not already set) would freeze
            # on the FIRST secret value seen and never notice you rotated
            # the key afterward, unless the app happened to fully reboot.
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # no secrets.toml present locally -- fine, the app will prompt for a key below

# ---------------------------------------------------------------- session --
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ------------------------------------------------------------------ header --
if LOGO_B64:
    st.markdown(f"""
    <div class="app-header">
        <img src="data:image/png;base64,{LOGO_B64}" class="app-logo" alt="cube logo" />
        <div>
            <h1 class="app-title">VendorScope</h1>
            <p class="app-tagline"><strong>Agentic RFP Evaluation &amp; Supplier Ranking.</strong>
            LLM judges proposal content only. All arithmetic, benchmarking,
            tie-breaks, and ranking are computed by deterministic Python.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.title("VendorScope")
    st.caption(
        "Agentic RFP Evaluation & Supplier Ranking. "
        "LLM judges proposal content only. All arithmetic, benchmarking, "
        "tie-breaks, and ranking are computed by deterministic Python."
    )

with st.sidebar:
    if LOGO_B64:
        st.markdown(
            f'<img src="data:image/png;base64,{LOGO_B64}" style="width:48px;margin-bottom:8px;" />',
            unsafe_allow_html=True,
        )
    st.header("Settings")

    engine = st.radio(
        "Pipeline engine",
        options=["Direct", "LangGraph"],
        index=0,
        help="Direct = plain Python orchestrator (agents/orchestrator.py). "
             "LangGraph = the same 10 steps modeled as an explicit typed-state "
             "graph (agents_langgraph/langgraph_pipeline.py): a start_batch "
             "node, an evaluate node that loops once per supplier (the ONLY "
             "node that calls an LLM), then benchmark_and_rank and persist "
             "nodes -- pure deterministic Python. Both engines always make a "
             "real LLM call and require an API key.",
    )
    if engine == "LangGraph":
        st.caption("Runs a compiled StateGraph: evaluate (loops per supplier) "
                    "-> benchmark_and_rank -> persist. Same pipeline, modeled "
                    "as an explicit graph instead of plain function calls.")

    provider_choice = st.selectbox(
        "Provider",
        options=["OpenRouter (recommended)", "Anthropic", "OpenAI"],
        help="OpenRouter needs one key and supports many models, "
             "including free-tier options -- easiest for a class project.",
    )
    key_env_map = {
        "OpenRouter (recommended)": ("OPENROUTER_API_KEY", "openrouter"),
        "Anthropic": ("ANTHROPIC_API_KEY", "anthropic"),
        "OpenAI": ("OPENAI_API_KEY", "openai"),
    }
    target_env, provider_hint = key_env_map[provider_choice]

    # A key present in os.environ here can ONLY have gotten there from
    # Streamlit secrets (mirrored once at startup, above) -- i.e. the app
    # owner's own key, deliberately shared with every visitor. User-typed
    # keys below are NEVER written to os.environ, specifically so they
    # can't leak across concurrent sessions on a shared deployment (see
    # session_api_key below).
    owner_key_configured = bool(os.environ.get(target_env))

    if "session_keys" not in st.session_state:
        st.session_state.session_keys = {}

    session_api_key = None
    if owner_key_configured:
        st.success(f"{target_env} is configured by the app owner (Streamlit secrets).")
    else:
        entered_key = st.text_input(
            f"{target_env}", type="password",
            help="Used only for YOUR current browser session. Never written "
                 "to the shared server environment, never visible to other "
                 "visitors. Only paste your own key into an app you trust -- "
                 "if you're the one deploying this publicly, prefer setting "
                 "it in Streamlit's Secrets panel instead, so visitors never "
                 "need to enter one at all.",
        )
        if entered_key:
            st.session_state.session_keys[target_env] = entered_key
            st.success(f"{target_env} set for this browser session only.")
        else:
            st.warning(f"Enter a key for this session, or ask the app owner "
                        f"to set {target_env} in Streamlit secrets.")
        session_api_key = st.session_state.session_keys.get(target_env)

    has_key = owner_key_configured or bool(session_api_key)

    with st.expander("Advanced (optional)"):
        model_choice = st.text_input(
            "Model override (optional)",
            value="",
            placeholder="e.g. meta-llama/llama-3.3-70b-instruct:free",
            help="Leave blank to use the provider's default model "
                 "(openai/gpt-4o-mini for OpenRouter). OpenRouter's default "
                 "model is a PAID one billed against your OpenRouter credit "
                 "balance -- if you're getting an APIStatusError / 402 on a "
                 "fresh account with $0 balance, that's almost certainly why. "
                 "Paste in a genuine free-tier model (any ':free'-suffixed "
                 "model from openrouter.ai/models) here to test with zero "
                 "cost instead.",
        )
        max_tokens_choice = st.number_input(
            "Max tokens per LLM call",
            min_value=50, max_value=2000, value=800, step=50,
            help="Caps the LLM's OUTPUT length for each supplier's scorecard. "
                 "Default (800) is enough for a normal 5-criterion response. "
                 "Lowering this deliberately (e.g. to 100-150) truncates the "
                 "model's JSON mid-response on a REAL call -- a reliable way "
                 "to demo tools/validation_tool.py's error-handling path "
                 "(missing criteria / parse-failure warnings) on live output, "
                 "rather than only via demo_validation_error_case.py's "
                 "hand-crafted example.",
        )
        if max_tokens_choice < 400:
            st.caption("⚠️ This is low enough to likely truncate a real response — "
                        "expect validation warnings on this run. Good for a demo, "
                        "not for a run you want clean results from.")

    st.divider()
    st.subheader("Past runs")
    runs = list_runs()
    if runs:
        run_labels = [f"{r['created_at'][:19]} — {r['status']} — {r['rfp_run_id'][:8]}" for r in runs]
        picked = st.selectbox("Reload a previous run", ["(none)"] + run_labels)
        if picked != "(none)":
            idx = run_labels.index(picked)
            st.session_state.last_result = load_run_from_db(runs[idx]["rfp_run_id"])
    else:
        st.caption("No runs yet.")

tab_criteria, tab_input, tab_leaderboard, tab_scorecard, tab_run = st.tabs(
    ["1. Criteria", "2. Supplier Input", "3. Leaderboard", "4. Detailed Scorecard", "5. Run Details"]
)

# ------------------------------------------------------------- 1. Criteria --
with tab_criteria:
    st.subheader("Active Evaluation Criteria")
    criteria = get_active_criteria()
    total_weight = active_weight_total()

    df_crit = pd.DataFrame(criteria)[["name", "description", "weight", "max_score"]]
    df_crit.columns = ["Criterion", "Description", "Weight (%)", "Max Score"]
    st.dataframe(df_crit, use_container_width=True, hide_index=True)

    if abs(total_weight - 100) > 0.001:
        st.error(f"Active criteria weights total {total_weight}%, not 100%. "
                  f"Fix this in the evaluation_criteria table before evaluating.")
    else:
        st.success(f"Active criteria weights total {total_weight}%. Ready to evaluate.")

# --------------------------------------------------------- 2. Supplier Input --
with tab_input:
    st.subheader("Upload Supplier Proposals")
    st.write("Upload one PDF per supplier and provide their metadata below.")

    uploaded_files = st.file_uploader(
        "Supplier RFP PDFs", type=["pdf"], accept_multiple_files=True
    )

    supplier_inputs = []
    if uploaded_files:
        st.write("Enter metadata for each uploaded proposal:")
        for f in uploaded_files:
            with st.expander(f"📄 {f.name}", expanded=True):
                c1, c2, c3 = st.columns(3)
                default_name = f.name.rsplit(".", 1)[0].replace("_", " ")
                name = c1.text_input("Supplier name", value=default_name, key=f"name_{f.name}")
                sub_date = c2.date_input("Submission date", key=f"date_{f.name}")
                exp_rating = c3.number_input(
                    "Historical experience rating (0-10)", min_value=0.0, max_value=10.0,
                    value=5.0, step=0.5, key=f"exp_{f.name}"
                )
                supplier_inputs.append({
                    "supplier_name": name.strip(),
                    "submission_date": sub_date.isoformat(),
                    "experience_rating": exp_rating,
                    "pdf_bytes": f.getvalue(),
                })

    # Validation messages before allowing evaluation
    validation_msgs = []
    names_seen = set()
    for s in supplier_inputs:
        if not s["supplier_name"]:
            validation_msgs.append("A supplier is missing a name.")
        if s["supplier_name"] in names_seen:
            validation_msgs.append(f'Duplicate supplier name: "{s["supplier_name"]}".')
        names_seen.add(s["supplier_name"])

    if len(supplier_inputs) < 2:
        validation_msgs.append("Upload at least 2 supplier proposals to enable peer benchmarking.")
    if abs(active_weight_total() - 100) > 0.001:
        validation_msgs.append("Active criteria weights do not total 100% — fix before evaluating.")
    if not has_key:
        validation_msgs.append("No API key is set — enter one in the sidebar first.")

    for m in validation_msgs:
        st.warning(m)

    can_evaluate = len(supplier_inputs) >= 2 and not any(
        "missing a name" in m or "Duplicate" in m or "do not total" in m
        or "No API key is set" in m for m in validation_msgs
    )

    if st.button("▶️ Evaluate Batch", type="primary", disabled=not can_evaluate):
        progress_bar = st.progress(0, text="Starting batch evaluation...")

        def on_progress(step, total, msg):
            progress_bar.progress(step / total, text=msg)

        # Only pass an explicit api_key/provider when using a session-scoped
        # key (i.e. the owner hasn't configured Streamlit secrets for this
        # provider). Passing None here for either falls through to
        # resolve_provider()'s environment-variable check, which will find
        # the owner's key mirrored in os.environ at startup, if present.
        eval_kwargs = {"api_key": session_api_key, "provider": provider_hint} \
            if not owner_key_configured else {}
        eval_kwargs["max_tokens"] = max_tokens_choice
        if model_choice.strip():
            eval_kwargs["model"] = model_choice.strip()

        try:
            with st.spinner(f"Running agentic evaluation workflow ({engine} engine)..."):
                if engine == "LangGraph":
                    result = run_langgraph_batch_evaluation(
                        supplier_inputs, get_active_criteria(),
                        progress_callback=on_progress, **eval_kwargs,
                    )
                else:
                    result = run_batch_evaluation(
                        supplier_inputs, get_active_criteria(),
                        progress_callback=on_progress, **eval_kwargs,
                    )
        except Exception as e:
            # Streamlit Cloud redacts unhandled-exception messages in the
            # UI (you only see "APIStatusError" with no detail, even though
            # the real message -- e.g. "insufficient_quota", an invalid
            # model name, or a bad key -- is sitting right there in the
            # exception). Catching it here and printing it directly means
            # you see the actual reason without needing repo/log access.
            progress_bar.empty()
            st.error(
                f"**Evaluation failed: `{type(e).__name__}`**\n\n"
                f"```\n{e}\n```\n\n"
                f"Common causes: an invalid/expired API key, an OpenRouter "
                f"account with insufficient credit balance for the selected "
                f"model (the default, `openai/gpt-4o-mini`, is a paid "
                f"model), or a model name that provider doesn't recognize. "
                f"See the Advanced settings above to try a different model."
            )
            st.stop()

        st.session_state.last_result = result
        progress_bar.progress(1.0, text="Done.")
        st.success(f"Batch complete. RFP_RUN_ID: {result['rfp_run_id']}")
        st.info("See the Leaderboard, Detailed Scorecard, and Run Details tabs.")

# --------------------------------------------------------- 3. Leaderboard --
RANK_STYLES = {
    1: {"color": "var(--gold)", "medal": "🥇", "bg": "#FFFBEF"},
    2: {"color": "var(--silver)", "medal": "🥈", "bg": "#F6F7F8"},
    3: {"color": "var(--bronze)", "medal": "🥉", "bg": "#FBF4EC"},
}


def render_podium(results):
    """Renders the top-3 ranked suppliers as accent cards -- the
    leaderboard's signature visual element. All user-controlled text
    (supplier_name) is HTML-escaped before interpolation, since this is
    rendered via unsafe_allow_html. Each card is built as a single-line
    string with no leading whitespace/newlines: Streamlit's markdown
    parser treats 4+ leading spaces as an indented code block, which would
    otherwise dump the HTML as raw text instead of rendering it."""
    top = sorted(results, key=lambda r: r["final_rank"])[:3]
    cards = []
    for r in top:
        style = RANK_STYLES.get(r["final_rank"],
                                 {"color": "var(--border)", "medal": f"#{r['final_rank']}", "bg": "var(--surface)"})
        safe_name = html.escape(r["supplier_name"])
        card = (
            f'<div class="podium-card" style="border-top-color:{style["color"]}; background:{style["bg"]};">'
            f'<div class="podium-medal">{style["medal"]}</div>'
            f'<div class="podium-rank">Rank {r["final_rank"]}</div>'
            f'<div class="podium-name">{safe_name}</div>'
            f'<div class="podium-ppi">PPI {r["ppi"]:.1f}</div>'
            f'</div>'
        )
        cards.append(card)
    st.markdown(f'<div class="podium-row">{"".join(cards)}</div>', unsafe_allow_html=True)


with tab_leaderboard:
    st.subheader("Supplier Leaderboard")
    result = st.session_state.last_result
    if not result:
        st.info("Run a batch evaluation (tab 2) or reload a past run (sidebar) to see results.")
    else:
        render_podium(result["results"])

        rows = []
        for r in result["results"]:
            rows.append({
                "Rank": r["final_rank"],
                "Supplier": r["supplier_name"],
                "Absolute Score": round(r["absolute_score"], 2),
                "PPI": round(r["ppi"], 2),
                "Submission Date": r["submission_date"],
                "Experience Rating": r["experience_rating"],
                "Warnings": len(r.get("warnings", [])),
            })
        df = pd.DataFrame(rows).sort_values("Rank")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Supplier")["PPI"])

# --------------------------------------------------- 4. Detailed Scorecard --
with tab_scorecard:
    st.subheader("Detailed Scorecard")
    result = st.session_state.last_result
    if not result:
        st.info("Run a batch evaluation (tab 2) or reload a past run (sidebar) to see scorecards.")
    else:
        supplier_names = [r["supplier_name"] for r in result["results"]]
        picked_supplier = st.selectbox("Choose a supplier", supplier_names)
        record = next(r for r in result["results"] if r["supplier_name"] == picked_supplier)

        c1, c2, c3 = st.columns(3)
        c1.metric("Final Rank", record["final_rank"])
        c2.metric("Absolute Score", round(record["absolute_score"], 2))
        c3.metric("PPI", round(record["ppi"], 2))

        crit_rows = []
        for c in record["criteria"]:
            crit_rows.append({
                "Criterion": c["name"],
                "Weight (%)": c["weight"],
                "Score": c["score"],
                "Max Score": c["max_score"],
                "Benchmark": c["benchmark"],
                "Gap": c["gap"],
                "Relative %": c["relative_pct"],
            })
        st.dataframe(pd.DataFrame(crit_rows), use_container_width=True, hide_index=True)

        st.markdown("**Evidence & Justification**")
        for c in record["criteria"]:
            with st.expander(f'{c["name"]} — score {c["score"]}/{c["max_score"]}'):
                st.write(f"**Justification:** {c['justification']}")
                st.write(f"**Evidence:** {c['evidence']}")

        if record.get("warnings"):
            st.markdown("**Validation warnings for this supplier**")
            for w in record["warnings"]:
                st.warning(w)

# --------------------------------------------------------------- 5. Run Details --
with tab_run:
    st.subheader("Run Details")
    result = st.session_state.last_result
    if not result:
        st.info("Run a batch evaluation (tab 2) or reload a past run (sidebar) to see run details.")
    else:
        st.write(f"**RFP_RUN_ID:** `{result['rfp_run_id']}`")
        st.write(f"**Created at:** {result['created_at']}")

        st.markdown("**Tie-break order applied**")
        st.markdown(
            "1. Higher PPI first\n"
            "2. Earlier submission date\n"
            "3. Higher historical experience rating\n"
            "4. Supplier name, ascending alphabetically\n\n"
            "Rank 1, 2, 3... assigned only after this stable sort."
        )

        st.markdown("**All warnings across suppliers**")
        any_warnings = False
        for r in result["results"]:
            for w in r.get("warnings", []):
                any_warnings = True
                st.warning(f"{r['supplier_name']}: {w}")
        if not any_warnings:
            st.success("No validation warnings were raised for this run.")

        st.markdown("**Download complete result**")
        json_bytes = json.dumps(result, indent=2, default=str).encode("utf-8")
        st.download_button(
            "⬇️ Download run as JSON",
            data=json_bytes,
            file_name=f"rfp_run_{result['rfp_run_id'][:8]}.json",
            mime="application/json",
        )
