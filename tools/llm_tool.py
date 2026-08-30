"""
llm_tool.py
Evaluation Agent: sends one supplier's proposal text + the active criteria
list to an LLM and asks for a strict-JSON, evidence-grounded scorecard.

The LLM is ONLY allowed to judge proposal content (scores, justification,
evidence, risks). It must NEVER compute weighted totals, benchmarks,
tie-breaks, or ranks -- that is done later by ranking_tool.py in pure
deterministic Python.

This module always makes a REAL LLM call. Provider resolution mirrors the
earlier AutoGen coding-agent project's convention:
  1. explicit api_key/base_url arguments
  2. OPENROUTER_API_KEY env var -> OpenRouter (recommended; one key, many
     models, free-tier options for a class project)
  3. ANTHROPIC_API_KEY env var -> direct Anthropic call
  4. OPENAI_API_KEY env var -> direct OpenAI call
If none of those resolve, resolve_provider() raises a clear ValueError --
there is no offline simulator to fall back to.

Robustness techniques (all stay within "LLM judges content, Python does
everything else" -- none of this changes what the LLM is allowed to
decide):
  - Structured output enforcement: OpenAI-compatible calls request
    response_format=json_schema (guarantees schema-valid JSON from
    providers that support it); Anthropic calls force a tool-use call
    against the same schema. Either way, tools/validation_tool.py still
    re-validates everything -- never trust the wire even with a schema.
  - Retry/backoff: transient rate-limit/server errors get up to 2 retries
    with exponential backoff before giving up.

Deliberately NOT implemented: self-consistency / majority voting (calling
the LLM 2-3x per supplier and taking the median score). It would triple
the per-supplier token/credit cost for a marginal robustness gain, which
cuts against this project's own token-budget goals -- noted here as a
technique considered and declined, not one worth building for this scope.
"""

import os
import json
import time

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_OPENROUTER = "openai/gpt-4o-mini"
DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_MODEL_ANTHROPIC = "claude-sonnet-4-6"
# max_tokens is capped to avoid OpenRouter 402 "insufficient credits"
# errors on free/low-balance accounts, which are triggered by the
# request's token budget, not actual usage.
DEFAULT_MAX_TOKENS = 800

MAX_RETRIES = 2          # additional attempts after the first, so 3 total
RETRY_BASE_DELAY = 1.5   # seconds; doubles each retry (1.5s, 3s)

# JSON Schema the LLM's scorecard must conform to. Used for structured
# output on OpenAI-compatible providers (response_format) and as the
# input_schema for a forced tool call on Anthropic. This is belt-and-
# braces with tools/validation_tool.py, not a replacement for it -- a
# schema constrains shape, not whether values are sensible, and providers
# that don't honor response_format still fall back to a plain call.
SCORECARD_SCHEMA = {
    "type": "object",
    "properties": {
        "supplier_name": {"type": "string"},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion_id": {"type": "integer"},
                    "score": {"type": "number"},
                    "max_score": {"type": "number"},
                    "justification": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["criterion_id", "score", "max_score", "justification", "evidence"],
            },
        },
        "risks": {"type": "array", "items": {"type": "string"}},
        "overall_summary": {"type": "string"},
    },
    "required": ["supplier_name", "criteria", "risks", "overall_summary"],
}


def build_prompt(supplier_name: str, proposal_text: str, criteria: list) -> str:
    criteria_lines = "\n".join(
        f'- criterion_id={c["criterion_id"]}, name="{c["name"]}", '
        f'max_score={c["max_score"]}, focus="{c["description"]}"'
        for c in criteria
    )
    return f"""You are an impartial procurement evaluator.

Supplier: {supplier_name}

Active evaluation criteria (you must return exactly one result for every
criterion listed, using the same criterion_id):
{criteria_lines}

Supplier proposal text (use ONLY evidence found in this text; do not invent
facts that are not present):
\"\"\"
{proposal_text[:12000]}
\"\"\"

Instructions:
1. Score each criterion from 0 to its max_score based only on evidence in
   the text above.
2. Provide a short justification and a direct quote/paraphrase as evidence
   for each score. The evidence must be text that actually appears in the
   proposal above -- do not paraphrase so loosely that it no longer
   matches the source wording.
3. List any notable risks you observe.
4. Provide a one or two sentence overall_summary.
5. Output ONLY valid JSON, no markdown fences, no commentary, matching
   exactly this schema:

{{
  "supplier_name": "string",
  "criteria": [
    {{"criterion_id": int, "score": number, "max_score": number,
      "justification": "string", "evidence": "string"}}
  ],
  "risks": ["string", ...],
  "overall_summary": "string"
}}
"""


def resolve_provider(model: str = None, api_key: str = None, base_url: str = None,
                      provider: str = None) -> dict:
    """
    Resolves which LLM backend a call should use, in priority order:
      1. explicit api_key (+ optional base_url / model / provider) passed by the caller
      2. OPENROUTER_API_KEY env var  -> OpenRouter, model defaults to
         "openai/gpt-4o-mini" (OpenRouter's provider-prefixed naming)
      3. ANTHROPIC_API_KEY env var   -> direct Anthropic call
      4. OPENAI_API_KEY env var      -> direct OpenAI call

    provider: only meaningful together with an explicit api_key -- hints
    which backend that key belongs to: "openrouter" (default if omitted),
    "anthropic", or "openai". Without this hint, an explicit api_key is
    assumed to be an OpenRouter key (this project's recommended provider),
    which would be wrong for an explicitly-Anthropic or -OpenAI key -- so
    any caller accepting a user-chosen provider alongside a user-entered
    key (e.g. a UI with a provider dropdown) should always pass this.

    Returns {"provider": "anthropic" | "openai_compatible", "api_key": ...,
             "model": ..., "base_url": ... (only for openai_compatible)}

    Raises ValueError if none resolve -- there is no fallback.
    """
    if api_key:
        if provider == "anthropic":
            return {
                "provider": "anthropic",
                "api_key": api_key,
                "model": model or DEFAULT_MODEL_ANTHROPIC,
            }
        if provider == "openai":
            return {
                "provider": "openai_compatible",
                "api_key": api_key,
                "base_url": base_url,  # None -> official OpenAI endpoint
                "model": model or DEFAULT_MODEL_OPENAI,
            }
        # provider == "openrouter" or unspecified: default assumption
        return {
            "provider": "openai_compatible",
            "api_key": api_key,
            "base_url": base_url or OPENROUTER_BASE_URL,
            "model": model or DEFAULT_MODEL_OPENROUTER,
        }

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        return {
            "provider": "openai_compatible",
            "api_key": openrouter_key,
            "base_url": base_url or os.environ.get("OPENAI_API_BASE") or OPENROUTER_BASE_URL,
            "model": model or DEFAULT_MODEL_OPENROUTER,
        }

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        return {
            "provider": "anthropic",
            "api_key": anthropic_key,
            "model": model or DEFAULT_MODEL_ANTHROPIC,
        }

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return {
            "provider": "openai_compatible",
            "api_key": openai_key,
            "base_url": base_url or os.environ.get("OPENAI_API_BASE"),  # None -> official OpenAI endpoint
            "model": model or DEFAULT_MODEL_OPENAI,
        }

    raise ValueError(
        "No LLM API key found. Set one of: OPENROUTER_API_KEY "
        "(recommended -- one key, many models), ANTHROPIC_API_KEY, or "
        "OPENAI_API_KEY as an environment variable, or pass api_key=... "
        "explicitly."
    )


def _is_retryable(exc: Exception) -> bool:
    """True for rate-limit / transient-server errors worth retrying.
    Checked defensively (by status code / class name) rather than
    importing every provider SDK's specific exception classes, since both
    openai and anthropic expose slightly different hierarchies across
    versions.

    IMPORTANT: only matches on the SPECIFIC subclass names for rate-limit
    and server errors -- never on the generic base class ("APIStatusError"
    for openai, "APIError" for anthropic). That base class is what gets
    raised for errors that have no more specific subclass -- including
    401 (bad key), 402 (insufficient credits/payment required), 403
    (forbidden), and 404 (model not found). None of those are fixed by
    retrying; matching the base class name here would silently waste time
    retrying a permanent failure before it finally surfaces to the caller.
    """
    status = getattr(exc, "status_code", None)
    if status in (429, 500, 502, 503, 529):
        return True
    name = type(exc).__name__
    return "RateLimitError" in name or "InternalServerError" in name or "APIConnectionError" in name


def _with_retries(fn, *args, **kwargs):
    """Runs fn(*args, **kwargs) with up to MAX_RETRIES extra attempts on
    retryable errors, using exponential backoff. Non-retryable errors
    (bad API key, malformed request, etc.) raise immediately."""
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt >= MAX_RETRIES or not _is_retryable(exc):
                raise
            time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
            attempt += 1


def _call_anthropic(cfg: dict, prompt: str, tokens: int) -> str:
    """Forces a structured tool-use call so Anthropic returns the
    scorecard as already-parsed, schema-shaped arguments rather than
    free-form text. tools/validation_tool.py still re-validates the
    result -- this only reduces malformed-JSON noise at the source."""
    import anthropic
    client = anthropic.Anthropic(api_key=cfg["api_key"])

    def _do_call():
        return client.messages.create(
            model=cfg["model"],
            max_tokens=tokens,
            tools=[{
                "name": "submit_scorecard",
                "description": "Submit the completed supplier evaluation scorecard.",
                "input_schema": SCORECARD_SCHEMA,
            }],
            tool_choice={"type": "tool", "name": "submit_scorecard"},
            messages=[{"role": "user", "content": prompt}],
        )

    response = _with_retries(_do_call)
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_scorecard":
            return json.dumps(block.input)
    # Fallback: no tool_use block found (shouldn't happen with a forced
    # tool_choice, but don't crash if a future API version changes shape) --
    # return whatever text came back so validation_tool can attempt to
    # parse/flag it through the normal error-handling path.
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")


def _call_openai_compatible(cfg: dict, prompt: str, tokens: int) -> str:
    """Requests structured JSON output via response_format when the
    provider/model supports it, and transparently falls back to a plain
    call if the provider rejects that parameter (not every OpenRouter
    backend model honors response_format)."""
    from openai import OpenAI
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg.get("base_url"))
    extra_headers = None
    if cfg.get("base_url") == OPENROUTER_BASE_URL:
        extra_headers = {"X-Title": "VendorScope"}

    def _do_call(use_schema: bool):
        kwargs = dict(
            model=cfg["model"],
            max_tokens=tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            extra_headers=extra_headers,
        )
        if use_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "scorecard", "schema": SCORECARD_SCHEMA, "strict": False},
            }
        return client.chat.completions.create(**kwargs)

    try:
        response = _with_retries(_do_call, use_schema=True)
    except Exception:
        # Provider/model doesn't support response_format (or some other
        # request-shape issue) -- retry once, plainly, before giving up.
        response = _with_retries(_do_call, use_schema=False)
    return response.choices[0].message.content


def call_llm_live(prompt: str, model: str = None, api_key: str = None,
                   base_url: str = None, max_tokens: int = None,
                   provider: str = None) -> str:
    """Makes one real LLM call and returns the raw text response.
    max_tokens caps the OUTPUT length of this single call (defaults to
    DEFAULT_MAX_TOKENS=800, comfortably enough for a 5-criterion JSON
    scorecard with short justifications) -- lower it further if you're on
    a tight token/credit budget, e.g. max_tokens=500. provider is only
    meaningful together with an explicit api_key -- see resolve_provider().
    """
    cfg = resolve_provider(model=model, api_key=api_key, base_url=base_url, provider=provider)
    tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS

    if cfg["provider"] == "anthropic":
        return _call_anthropic(cfg, prompt, tokens)
    return _call_openai_compatible(cfg, prompt, tokens)


def evaluate_supplier(supplier_name: str, proposal_text: str, criteria: list,
                       model: str = None, api_key: str = None,
                       base_url: str = None, max_tokens: int = None,
                       provider: str = None) -> str:
    """
    Builds the evaluation prompt and makes a real LLM call, returning the
    raw JSON string response (not yet parsed/validated -- that's
    tools/validation_tool.py's job). model/api_key/base_url/max_tokens/
    provider are passed straight to resolve_provider()/call_llm_live();
    leave them None to resolve purely from environment variables. Raises
    ValueError if no API key resolves.
    """
    prompt = build_prompt(supplier_name, proposal_text, criteria)
    return call_llm_live(prompt, model=model, api_key=api_key,
                          base_url=base_url, max_tokens=max_tokens, provider=provider)
