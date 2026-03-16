from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import AsyncAzureOpenAI
from datetime import datetime
import uuid
import logging
import json
import re

from app.database import ChatSessions
from app.config import settings
from app.constant.dashboard import dashboard_data

logger = logging.getLogger(__name__)

router = APIRouter()
openai_client = AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)


# ---------------------------------------------------------------------------
# Build compact dashboard context (injected into every system prompt)
# ---------------------------------------------------------------------------

def _build_dashboard_context() -> str:
    """
    Extract a compact, token-efficient representation of the full dashboard
    dataset to inject into the AI system prompt.
    """
    metadata = dashboard_data.get("metadata", {})
    portfolio = dashboard_data.get("portfolio_summary", {})
    projects = dashboard_data.get("projects", [])

    compact_projects = []
    for p in projects:
        stages_summary = [
            {
                "stage": s.get("stage_number"),
                "name": s.get("name"),
                "authority": s.get("authority"),
                "status": s.get("status"),
                "sla_days": s.get("sla_days"),
                "actual_days": s.get("actual_days"),
                "days_elapsed": s.get("days_elapsed"),
                "days_overdue": s.get("days_overdue"),
                "notes": s.get("notes", "")[:120],
            }
            for s in p.get("stages", [])
        ]
        noc_tracker_summary = [
            {
                "authority": n.get("authority"),
                "type": n.get("type"),
                "status": n.get("status"),
                "sla_days": n.get("sla_days"),
                "days_taken": n.get("days_taken"),
                "days_elapsed": n.get("days_elapsed"),
            }
            for n in p.get("noc_tracker", [])
        ]
        compact_projects.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "type": p.get("type"),
            "emirate": p.get("emirate"),
            "capacity_mw": p.get("capacity_mw"),
            "value_aed": p.get("value_aed"),
            "developer": p.get("developer"),
            "status": p.get("status"),
            "current_stage": p.get("current_stage"),
            "readiness_score": p.get("readiness_score"),
            "days_since_last_update": p.get("days_since_last_update"),
            "last_update_note": p.get("last_update_note", "")[:150],
            "noc_summary": p.get("noc_summary", {}),
            "noc_tracker": noc_tracker_summary,
            "ai_flags": [
                {
                    "severity": f.get("severity"),
                    "title": f.get("title"),
                    "description": f.get("description", "")[:120],
                    "recommendation": f.get("recommendation", "")[:120],
                }
                for f in p.get("ai_flags", [])
            ],
            "documents": p.get("documents", {}),
            "stages": stages_summary,
        })

    context = {
        "metadata": metadata,
        "portfolio_summary": portfolio,
        "projects": compact_projects,
    }
    return json.dumps(context, indent=2)


DASHBOARD_CONTEXT = _build_dashboard_context()  # built once at startup


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a **Dashboard Intelligence Agent** and **AI Permit Readiness Assistant for UAE Energy Project Applications**.
You have full access to the live portfolio dashboard data provided below.
Use it to answer every question with precise, data-driven analysis.

=== DASHBOARD DATA ===
{DASHBOARD_CONTEXT}
=== END DASHBOARD DATA ===

RESPONSE FORMAT — CRITICAL:
You MUST always respond with a single valid JSON object containing exactly these two keys:

{{
  "markdown": <markdown string or null>,
  "html": <HTML string or null>
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — ANALYSE THE USER QUERY FIRST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before writing any response, classify the user's intent:

  A) Does the user want a TEXT ANALYSIS / explanation / summary?
     → Set "markdown" to a full analysis (not null)

  B) Does the user explicitly ask for a CHART, GRAPH, or VISUALIZATION?
     → Set "html" to a Chart.js snippet (not null)

  C) Does the user want BOTH (e.g. "analyse and show a chart")?
     → Set both fields

  D) Does the user want ONLY a chart with no explanation?
     → Set "markdown" to null, "html" to the chart

  E) Does the user want ONLY a text answer (no chart mentioned)?
     → Set "html" to null, "markdown" to the analysis

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — RULES FOR EACH FIELD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — "markdown" field:
  - If not needed → return JSON null (not "null", not "", not a string)
  - If needed → thorough GitHub-flavoured markdown using:
      • ## and ### headers
      • **Bold** for project names, numbers, risk labels
      • Bullet and numbered lists
      • Markdown tables for comparisons and data summaries
      • Emojis sparingly: 🔴 Blocked · 🟡 At Risk · 🟢 On Track
  - Always include a data table when presenting comparisons,
    scores, counts, or distributions — even alongside a chart

RULE 2 — "html" field:
  - If not needed → return JSON null (not "null", not "", not a div)
  - Only generate HTML when the user explicitly requests a chart,
    graph, visualization, or diagram
  - When generating HTML:
      • Always use the unique canvas ID from the user message (format: chart_<uuid>)
      • Always load Chart.js from CDN:
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
      • Wrap everything in:
        <div style="width:100%;max-width:700px;margin:0 auto;font-family:sans-serif;">
      • Pick the best chart type:
          - Status distribution  → doughnut or pie
          - Readiness scores     → horizontal bar
          - NOC counts           → stacked bar
          - Stage progress       → grouped bar
          - Trends / timeline    → line chart
      • Use these colours consistently:
          - On Track        #22c55e   (green)
          - At Risk         #f59e0b   (amber)
          - Blocked         #ef4444   (red)
          - Neutral series  #3b82f6 · #8b5cf6 · #06b6d4 · #f97316
      • Set responsive: true, maintainAspectRatio: true
      • Add a descriptive title via the Chart.js title plugin

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES OF CORRECT CLASSIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "Which projects are blocked?"
    → markdown: full analysis with table  |  html: null

  "Show me a chart of project statuses"
    → markdown: null  |  html: chart snippet

  "Analyse readiness scores and show a bar chart"
    → markdown: analysis + table  |  html: bar chart

  "What is the overall portfolio health?"
    → markdown: full summary  |  html: null

  "Give me a pie chart of NOC approval rates"
    → markdown: null  |  html: pie chart

  "hello" or "hi" or vague message
    → markdown: brief prompt asking what they need  |  html: null

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCOPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only answer questions about the dashboard data:
project status · permits · NOCs · readiness scores · risk analysis ·
SLA performance · stage progress · authority delays · AI flags ·
documents · capacity · portfolio value

Decline anything outside this scope.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT OUTPUT RULES — NEVER BREAK THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- NEVER return a greeting, filler, or placeholder text in "markdown"
- NEVER return a placeholder div or empty div in "html" — use null
- NEVER return "No visualisation available" in "html" — use null
- NEVER return an empty string "" for either field — use null
- NEVER wrap null in quotes — it must be JSON null, not the string "null"
- "html" must be null unless the user explicitly asks for a chart,
  graph, or visualization in their message
- If the user sends a greeting or vague message with no clear query,
  respond in "markdown" with a brief prompt asking what they need,
  and set "html" to null"""


GREETING_MESSAGE = "Hello! I'm your Dashboard Intelligence Agent for UAE Energy Project Applications. How can I assist you today?"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SessionResponse(BaseModel):
    session_id: str
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    message: str
    markdown: Optional[str] = None
    html: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_session(session_id: str) -> dict:
    session = await ChatSessions.find_one({"session_id": session_id, "type": "dashboard"})
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Dashboard session not found. Call POST /api/dashboard/chat/session first.",
        )
    return session


def _extract_json(raw: str) -> dict:
    """
    Robustly extract JSON from the AI response.
    Handles cases where the model wraps JSON in markdown code fences.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find the first { ... } block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # Fallback — return markdown only, html as None
    return {
        "markdown": raw,
        "html": None,
    }


def _sanitise_field(value) -> Optional[str]:
    """
    Normalise a field returned by the model:
    - JSON null / Python None / missing  → None
    - Empty string or whitespace-only    → None
    - Placeholder div strings            → None
    - Anything else                      → stripped string
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    # Reject placeholder patterns the model sometimes returns
    placeholder_patterns = [
        "no visuali",        # "No visualisation available…"
        "no chart",
        "no graph",
        "<div style='padding:16px",
        '<div style="padding:16px',
    ]
    lower = stripped.lower()
    if any(p in lower for p in placeholder_patterns):
        return None
    return stripped


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/session", response_model=SessionResponse)
async def create_dashboard_session():
    """
    Create a new dashboard chat session.
    Call this when the dashboard chat panel opens.
    """
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()

    await ChatSessions.insert_one({
        "session_id": session_id,
        "type": "dashboard",
        "created_at": now,
        "messages": [
            {
                "role": "assistant",
                "content": GREETING_MESSAGE,
                "timestamp": now.isoformat(),
            }
        ],
    })

    return SessionResponse(session_id=session_id, message=GREETING_MESSAGE)


@router.post("/message", response_model=ChatResponse)
async def send_dashboard_message(body: ChatRequest):
    """
    Send a question to the dashboard agent.

    Returns a structured JSON response with:
    - message  : plain-text summary (for chat bubble)
    - markdown : detailed markdown analysis, or null
    - html     : self-contained Chart.js HTML snippet, or null
    """
    session = await _get_session(body.session_id)
    messages_history: list = session.get("messages", [])

    # Generate a unique chart ID for this response to avoid DOM conflicts
    chart_id = f"chart_{uuid.uuid4().hex[:12]}"

    # Build OpenAI message list from history (last 10 turns to stay within token budget)
    openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages_history[-10:]:
        openai_messages.append({"role": msg["role"], "content": msg["content"]})

    # Append user message with the chart_id hint
    user_content = (
        f"{body.message}\n\n"
        f"[Use canvas id=\"{chart_id}\" for your Chart.js chart in the html field if a chart is needed]"
    )
    openai_messages.append({"role": "user", "content": user_content})

    try:
        response = await openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_GPT4O,
            messages=openai_messages,
            temperature=0.2,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content or "{}"
    except Exception as e:
        logger.error("Azure OpenAI dashboard chat error for session %s: %s", body.session_id, e)
        raise HTTPException(status_code=500, detail="Failed to get a response from AI. Please try again.")

    parsed = _extract_json(raw_content)

    # Sanitise both fields — converts placeholders / empty strings to None
    markdown_text = _sanitise_field(parsed.get("markdown"))
    html_text     = _sanitise_field(parsed.get("html"))

    # Derive a plain-text message for the chat bubble
    if markdown_text:
        # First non-empty line of markdown, stripped of markdown syntax
        first_line = re.sub(r"[#*`_>~]", "", markdown_text.splitlines()[0]).strip()
        message_text = first_line or "Here is the analysis based on the dashboard data."
    else:
        message_text = "Here is the chart based on your request." if html_text else "Here is the response."

    now = datetime.utcnow()

    # Persist conversation
    await ChatSessions.update_one(
        {"session_id": body.session_id},
        {
            "$push": {
                "messages": {
                    "$each": [
                        {
                            "role": "user",
                            "content": body.message,
                            "timestamp": now.isoformat(),
                        },
                        {
                            "role": "assistant",
                            "content": message_text,
                            "timestamp": now.isoformat(),
                        },
                    ]
                }
            }
        },
    )

    return ChatResponse(
        session_id=body.session_id,
        message=message_text,
        markdown=markdown_text,
        html=html_text,
    )


@router.get("/session/{session_id}/history")
async def get_dashboard_session_history(session_id: str):
    """Return the full message history for a dashboard chat session."""
    session = await _get_session(session_id)
    created_at = session.get("created_at")
    return {
        "session_id": session_id,
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
        "messages": session.get("messages", []),
    }










# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import Optional
# from openai import AsyncAzureOpenAI
# from datetime import datetime
# import uuid
# import logging
# import json
# import re

# from app.database import ChatSessions
# from app.config import settings
# from app.constant.dashboard import dashboard_data

# logger = logging.getLogger(__name__)

# router = APIRouter()
# openai_client = AsyncAzureOpenAI(
#     api_key=settings.AZURE_OPENAI_API_KEY,
#     azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
#     api_version=settings.AZURE_OPENAI_API_VERSION,
# )


# # ---------------------------------------------------------------------------
# # Build compact dashboard context (injected into every system prompt)
# # ---------------------------------------------------------------------------

# def _build_dashboard_context() -> str:
#     """
#     Extract a compact, token-efficient representation of the full dashboard
#     dataset to inject into the AI system prompt.
#     """
#     metadata = dashboard_data.get("metadata", {})
#     portfolio = dashboard_data.get("portfolio_summary", {})
#     projects = dashboard_data.get("projects", [])

#     compact_projects = []
#     for p in projects:
#         stages_summary = [
#             {
#                 "stage": s.get("stage_number"),
#                 "name": s.get("name"),
#                 "authority": s.get("authority"),
#                 "status": s.get("status"),
#                 "sla_days": s.get("sla_days"),
#                 "actual_days": s.get("actual_days"),
#                 "days_elapsed": s.get("days_elapsed"),
#                 "days_overdue": s.get("days_overdue"),
#                 "notes": s.get("notes", "")[:120],
#             }
#             for s in p.get("stages", [])
#         ]
#         noc_tracker_summary = [
#             {
#                 "authority": n.get("authority"),
#                 "type": n.get("type"),
#                 "status": n.get("status"),
#                 "sla_days": n.get("sla_days"),
#                 "days_taken": n.get("days_taken"),
#                 "days_elapsed": n.get("days_elapsed"),
#             }
#             for n in p.get("noc_tracker", [])
#         ]
#         compact_projects.append({
#             "id": p.get("id"),
#             "name": p.get("name"),
#             "type": p.get("type"),
#             "emirate": p.get("emirate"),
#             "capacity_mw": p.get("capacity_mw"),
#             "value_aed": p.get("value_aed"),
#             "developer": p.get("developer"),
#             "status": p.get("status"),
#             "current_stage": p.get("current_stage"),
#             "readiness_score": p.get("readiness_score"),
#             "days_since_last_update": p.get("days_since_last_update"),
#             "last_update_note": p.get("last_update_note", "")[:150],
#             "noc_summary": p.get("noc_summary", {}),
#             "noc_tracker": noc_tracker_summary,
#             "ai_flags": [
#                 {
#                     "severity": f.get("severity"),
#                     "title": f.get("title"),
#                     "description": f.get("description", "")[:120],
#                     "recommendation": f.get("recommendation", "")[:120],
#                 }
#                 for f in p.get("ai_flags", [])
#             ],
#             "documents": p.get("documents", {}),
#             "stages": stages_summary,
#         })

#     context = {
#         "metadata": metadata,
#         "portfolio_summary": portfolio,
#         "projects": compact_projects,
#     }
#     return json.dumps(context, indent=2)


# DASHBOARD_CONTEXT = _build_dashboard_context()   # built once at startup


# # ---------------------------------------------------------------------------
# # System prompt
# # ---------------------------------------------------------------------------

# SYSTEM_PROMPT = f"""You are a **Dashboard Intelligence Agent** and **AI Permit Readiness Assistant for UAE Energy Project Applications**. 
# You have full access to the live portfolio dashboard data provided below. 
# Use it to answer every question with precise, data-driven analysis.

# === DASHBOARD DATA ===
# {DASHBOARD_CONTEXT}
# === END DASHBOARD DATA ===

# RESPONSE FORMAT — CRITICAL:
# You MUST always respond with a single valid JSON object containing exactly these two keys:

# {{
#   "markdown": "<markdown analysis or null>",
#   "html": "<Chart.js HTML snippet or null>"
# }}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — ANALYSE THE USER QUERY FIRST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Before writing any response, classify the user's intent:

#   A) Does the user want a TEXT ANALYSIS / explanation / summary?
#      → Set "markdown" to a full analysis (not null)

#   B) Does the user explicitly ask for a CHART, GRAPH, or VISUALIZATION?
#      → Set "html" to a Chart.js snippet (not null)

#   C) Does the user want BOTH (e.g. "analyse and show a chart")?
#      → Set both fields

#   D) Does the user want ONLY a chart with no explanation?
#      → Set "markdown" to null, "html" to the chart

#   E) Does the user want ONLY a text answer (no chart mentioned)?
#      → Set "html" to null, "markdown" to the analysis

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — RULES FOR EACH FIELD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# RULE 1 — "markdown" field:
#   - If not needed → return null exactly (no quotes, no empty string)
#   - If needed → thorough GitHub-flavoured markdown using:
#       • ## and ### headers
#       • **Bold** for project names, numbers, risk labels
#       • Bullet and numbered lists
#       • Markdown tables for comparisons and data summaries
#       • Emojis sparingly: 🔴 Blocked · 🟡 At Risk · 🟢 On Track
#   - Always include a data table when presenting comparisons,
#     scores, counts, or distributions — even alongside a chart

# RULE 2 — "html" field:
#   - If not needed → return null exactly (no quotes, no empty string)
#   - Only generate HTML when the user explicitly requests a chart,
#     graph, visualization, or diagram
#   - When generating HTML:
#       • Always use the unique canvas ID from the user message (format: chart_<uuid>)
#       • Always load Chart.js from CDN:
#         <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
#       • Wrap everything in:
#         <div style="width:100%;max-width:700px;margin:0 auto;font-family:sans-serif;">
#       • Pick the best chart type:
#           - Status distribution  → doughnut or pie
#           - Readiness scores     → horizontal bar
#           - NOC counts           → stacked bar
#           - Stage progress       → grouped bar
#           - Trends / timeline    → line chart
#       • Use these colours consistently:
#           - On Track        #22c55e   (green)
#           - At Risk         #f59e0b   (amber)
#           - Blocked         #ef4444   (red)
#           - Neutral series  #3b82f6 · #8b5cf6 · #06b6d4 · #f97316
#       • Set responsive: true, maintainAspectRatio: true
#       • Add a descriptive title via the Chart.js title plugin

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EXAMPLES OF CORRECT CLASSIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   "Which projects are blocked?"
#     → markdown: full analysis with table  |  html: null

#   "Show me a chart of project statuses"
#     → markdown: null  |  html: chart snippet

#   "Analyse readiness scores and show a bar chart"
#     → markdown: analysis + table  |  html: bar chart

#   "What is the overall portfolio health?"
#     → markdown: full summary  |  html: null

#   "Give me a pie chart of NOC approval rates"
#     → markdown: null  |  html: pie chart

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SCOPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Only answer questions about the dashboard data:
# project status · permits · NOCs · readiness scores · risk analysis ·
# SLA performance · stage progress · authority delays · AI flags ·
# documents · capacity · portfolio value

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STRICT OUTPUT RULES — NEVER BREAK THESE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# - NEVER return a greeting, filler, or placeholder text in "markdown"
# - NEVER return a placeholder div or empty div in "html" — use null
# - NEVER return "No visualisation available" in "html" — use null
# - NEVER return an empty string "" for either field — use null
# - If the user sends a greeting or vague message with no clear query,
#   respond in "markdown" with a brief prompt asking what they need,
#   and set "html" to null
# - "html" must be null unless the user explicitly asks for a chart,
#   graph, or visualization in their message
# - null must be unquoted JSON null — not "null", not "", not a div

# Decline anything outside this scope."""


# GREETING_MESSAGE = (
#     "Hello! I'm the Dashboard Intelligence Agent.\n\n"
# )


# # ---------------------------------------------------------------------------
# # Pydantic models
# # ---------------------------------------------------------------------------

# class SessionResponse(BaseModel):
#     session_id: str
#     message: str


# class ChatRequest(BaseModel):
#     session_id: str
#     message: str


# class ChatResponse(BaseModel):
#     session_id: str
#     message: str
#     markdown: str
#     html: str


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# async def _get_session(session_id: str) -> dict:
#     session = await ChatSessions.find_one({"session_id": session_id, "type": "dashboard"})
#     if not session:
#         raise HTTPException(
#             status_code=404,
#             detail="Dashboard session not found. Call POST /api/dashboard/chat/session first.",
#         )
#     return session


# def _extract_json(raw: str) -> dict:
#     """
#     Robustly extract JSON from the AI response.
#     Handles cases where the model wraps JSON in markdown code fences.
#     """
#     # Strip markdown code fences if present
#     cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
#     cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

#     try:
#         return json.loads(cleaned)
#     except json.JSONDecodeError:
#         # Try to find the first { ... } block
#         match = re.search(r"\{.*\}", cleaned, re.DOTALL)
#         if match:
#             try:
#                 return json.loads(match.group())
#             except json.JSONDecodeError:
#                 pass

#     # Fallback — return as plain message
#     return {
#         "message": raw[:300],
#         "markdown": raw,
#         "html": "<div style='padding:16px;color:#6b7280;font-family:sans-serif;'>No visualisation available.</div>",
#     }


# # ---------------------------------------------------------------------------
# # Endpoints
# # ---------------------------------------------------------------------------

# @router.post("/session", response_model=SessionResponse)
# async def create_dashboard_session():
#     """
#     Create a new dashboard chat session.
#     Call this when the dashboard chat panel opens.
#     """
#     session_id = str(uuid.uuid4())
#     now = datetime.utcnow()

#     await ChatSessions.insert_one({
#         "session_id": session_id,
#         "type": "dashboard",
#         "created_at": now,
#         "messages": [
#             {
#                 "role": "assistant",
#                 "content": GREETING_MESSAGE,
#                 "timestamp": now.isoformat(),
#             }
#         ],
#     })

#     return SessionResponse(session_id=session_id, message=GREETING_MESSAGE)


# @router.post("/message", response_model=ChatResponse)
# async def send_dashboard_message(body: ChatRequest):
#     """
#     Send a question to the dashboard agent.

#     Returns a structured JSON response with:
#     - message   : plain-text summary (for chat bubble)
#     - markdown  : detailed markdown analysis
#     - html      : self-contained Chart.js HTML snippet ready to inject into the frontend
#     """
#     session = await _get_session(body.session_id)
#     messages_history: list = session.get("messages", [])

#     # Generate a unique chart ID for this response to avoid DOM conflicts
#     chart_id = f"chart_{uuid.uuid4().hex[:12]}"

#     # Build OpenAI message list from history (last 10 turns to stay within token budget)
#     openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
#     for msg in messages_history[-10:]:
#         openai_messages.append({"role": msg["role"], "content": msg["content"]})

#     # Append user message with the chart_id hint
#     user_content = (
#         f"{body.message}\n\n"
#         f"[Use canvas id=\"{chart_id}\" for your Chart.js chart in the html field]"
#     )
#     openai_messages.append({"role": "user", "content": user_content})

#     try:
#         response = await openai_client.chat.completions.create(
#             model=settings.AZURE_OPENAI_DEPLOYMENT_GPT4O,
#             messages=openai_messages,
#             temperature=0.2,
#             max_tokens=2500,
#             response_format={"type": "json_object"},
#         )
#         raw_content = response.choices[0].message.content or "{}"
#     except Exception as e:
#         logger.error("Azure OpenAI dashboard chat error for session %s: %s", body.session_id, e)
#         raise HTTPException(status_code=500, detail="Failed to get a response from AI. Please try again.")

#     parsed = _extract_json(raw_content)

#     message_text = parsed.get("message", "")
#     markdown_text = parsed.get("markdown", "")
#     html_text = parsed.get("html", "")

#     # Fallback values if the model omitted a field
#     if not message_text:
#         message_text = "Here is the analysis based on the dashboard data."
#     if not markdown_text:
#         markdown_text = message_text
#     if not html_text:
#         html_text = "<div style='padding:16px;color:#6b7280;font-family:sans-serif;'>No visualisation available for this query.</div>"

#     now = datetime.utcnow()

#     # Persist conversation — store the raw user message (without chart_id hint) and assistant reply
#     await ChatSessions.update_one(
#         {"session_id": body.session_id},
#         {
#             "$push": {
#                 "messages": {
#                     "$each": [
#                         {
#                             "role": "user",
#                             "content": body.message,
#                             "timestamp": now.isoformat(),
#                         },
#                         {
#                             "role": "assistant",
#                             "content": message_text,
#                             "timestamp": now.isoformat(),
#                         },
#                     ]
#                 }
#             }
#         },
#     )

#     return ChatResponse(
#         session_id=body.session_id,
#         message=message_text,
#         markdown=markdown_text,
#         html=html_text,
#     )


# @router.get("/session/{session_id}/history")
# async def get_dashboard_session_history(session_id: str):
#     """Return the full message history for a dashboard chat session."""
#     session = await _get_session(session_id)
#     created_at = session.get("created_at")
#     return {
#         "session_id": session_id,
#         "created_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
#         "messages": session.get("messages", []),
#     }
