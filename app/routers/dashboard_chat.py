from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import AsyncAzureOpenAI
from datetime import datetime
import uuid
import logging
import json
import re
import os

from app.database import ChatSessions
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
openai_client = AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)


_here = os.path.dirname(__file__)

with open(os.path.join(_here, "../constant/dashboard.json"), "r") as f:
    DASHBOARD_CONTEXT = json.load(f)# built once at startup


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Role: AI Analytics Assistant for UAE Energy Project Permit Portfolio.

Objective:
Answer officer queries by analyzing the provided project portfolio data and generating a clear visual representation that helps decision-makers quickly understand the results.

Instructions:
1. Interpret the user's query.
2. Analyze and filter the provided JSON data accordingly.
3. Decide the most suitable way to present the result for easy understanding.
4. Generate a compact visual response using HTML markup.

Visualization Selection Guidelines:
Choose the representation that best communicates the result:
- Use KPI cards for key numbers or totals.
- Use tables when listing projects or detailed records.
- Use bar indicators for comparisons (capacity, counts, delays).
- Use grouped summaries for distribution (status, emirate, energy type).
- Highlight important conditions such as "At Risk", "Blocked", "Delayed".
- When appropriate, include a short insight summary at the top.

Formatting Rules:
- Use clean HTML with minimal inline CSS.
- Prefer simple dashboard components: cards, tables, lists, and horizontal bars.
- Ensure the layout is compact and readable within a chat or dashboard panel.

Data Integrity Rules:
- Use ONLY the provided JSON data.
- Do NOT fabricate values or fields.
- If the requested data is not available, clearly state it.

Output Rules:
- Return ONLY valid HTML markup.
- Do NOT include markdown or explanations.

Conversation Handling:
- If the user sends a greeting (e.g., hi, hello) or a non-analytics message, respond with a short greeting and explain what analytics questions can be asked.
- In this case return simple HTML text, not a dashboard.

Data:
{analytics_json}
"""


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
    openai_messages = [{"role": "system", "content": SYSTEM_PROMPT.format(analytics_json=json.dumps(DASHBOARD_CONTEXT, indent=2))}]
    for msg in messages_history[-10:]:
        openai_messages.append({"role": msg["role"], "content": msg["content"]})

    # Append user message
    openai_messages.append({"role": "user", "content": body.message})

    try:
        response = await openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_GPT4O,
            messages=openai_messages,
            temperature=0.2,
            max_tokens=2500,
        )
        raw_content = response.choices[0].message.content or ""
    except Exception as e:
        logger.error("Azure OpenAI dashboard chat error for session %s: %s", body.session_id, e)
        raise HTTPException(status_code=500, detail="Failed to get a response from AI. Please try again.")

    # With the new prompt, the output is pure HTML.
    # We remove any potential markdown code blocks if the AI decided to wrap it.
    html_text = re.sub(r"^```(?:html)?\s*", "", raw_content.strip(), flags=re.MULTILINE)
    html_text = re.sub(r"\s*```$", "", html_text.strip(), flags=re.MULTILINE)

    # Derive a plain-text message for the chat bubble (summary)
    summary_match = re.search(r"<(?:p|h[1-6]|div)[^>]*>(.*?)</(?:p|h[1-6]|div)>", html_text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        # Strip internal tags
        message_text = re.sub(r"<[^>]+>", "", summary_match.group(1)).strip()
        if len(message_text) > 200:
            message_text = message_text[:197] + "..."
    else:
        message_text = "Analysis complete. See visual report below."

    if not message_text:
        message_text = "Analysis complete. See visual report below."

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
        markdown=None,
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





