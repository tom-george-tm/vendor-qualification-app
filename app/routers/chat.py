from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import AsyncAzureOpenAI
from datetime import datetime
import uuid
import logging

from app.database import Results, ChatSessions
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
openai_client = AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

REQUIRED_DOCUMENTS = [
    "Environmental Impact Assessment (EIA)",
    "Grid Assessment Report",
    "EPC Contract",
    "GCAA No Objection Certificate (NOC)",
    "Bank Financing Letter",
]

GREETING_MESSAGE = (
    "Hello! Welcome to the Permit Assignment and Validation Assistant.\n\n"
)

SYSTEM_PROMPT = """You are a Permit Assignment and Validation Assistant specialising in permit validation and document analysis for energy infrastructure projects in the UAE.

Your role is to:
- Welcome users and guide them through the vendor qualification process
- Answer questions about permit requirements and what each document must contain
- Explain document analysis results including checklist findings, deviations, and risk levels
- Guide vendors on which documents are required (EIA, Grid Assessment, EPC Contract, GCAA NOC, Bank Letter) and why
- Provide actionable recommendations based on document analysis results when available
- Explain APPROVE/REJECT decisions and what the vendor must do to address deficiencies

When NO document analysis data is available yet:
- Respond naturally and helpfully to greetings, general questions, and process questions
- Guide the user toward uploading their 5 required documents (EIA, Grid Assessment, EPC Contract, GCAA NOC, Bank Letter)
- Explain what each document should contain, common checklist criteria, and what makes a document pass or fail
- Never refuse a greeting or a reasonable question about the qualification process

When document analysis data IS available (provided in the context block below):
- Ground every answer in the actual analysis data — cite specific documents, risk levels, checklist criteria, and scores
- Reference the readiness score, submission status, and per-document findings when relevant
- Use cross-document validation findings and action items to give precise recommendations

Only redirect the user if they ask about something completely unrelated to permits, document validation, vendor qualification, or energy infrastructure (e.g. cooking, sports, unrelated technical topics). In that case respond:
"I can only assist with permit validation and document analysis topics. Please ask me about your document submission, permit requirements, or qualification status."

Tone: professional, precise, and helpful."""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SessionResponse(BaseModel):
    session_id: str
    message: str
    has_analysis_data: bool


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    message: str
    has_analysis_data: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_doc(doc: dict) -> dict:
    """Recursively convert MongoDB doc to JSON-serialisable dict."""
    result = {}
    for k, v in doc.items():
        if k == "_id":
            result[k] = str(v)
        elif isinstance(v, datetime):
            result[k] = v.isoformat()
        elif isinstance(v, dict):
            result[k] = _serialize_doc(v)
        elif isinstance(v, list):
            result[k] = [_serialize_doc(i) if isinstance(i, dict) else i for i in v]
        else:
            result[k] = v
    return result


async def _get_session_analysis(session_id: str) -> Optional[dict]:
    """
    Fetch the most recent aggregated result that belongs to this session.
    Returns None if the session has not uploaded any documents yet.
    """
    doc = await Results.find_one(
        {"session_id": session_id},
        sort=[("created_at", -1)],
    )
    if not doc:
        return None
    return _serialize_doc(doc)


async def _get_session(session_id: str) -> dict:
    """Return the session document or raise 404."""
    session = await ChatSessions.find_one({"session_id": session_id})
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call POST /api/chat/session to start a new one.",
        )
    return session


def _build_context_block(analysis: Optional[dict]) -> str:
    """
    Convert the latest aggregated analysis for this session into a compact
    context block that is appended to the system prompt.
    """
    if not analysis:
        return (
            "\n\n--- DOCUMENT ANALYSIS CONTEXT ---\n"
            "STATUS: No documents uploaded for this session yet.\n"
            "BEHAVIOUR: Respond naturally to the user's message. Be conversational and helpful. "
            "If they greet you, greet them back and briefly explain what you can help with. "
            "If they ask about the process, explain it clearly. "
            "If they ask what documents are needed, list and explain each one: "
            "EIA (environmental baseline, marine impact, 18-month study date), "
            "Grid Assessment (load data 2023+, named substation, fault level analysis), "
            "EPC Contract (Arabic translation, performance bonds, signed by both parties), "
            "GCAA NOC (approval within 12 months, correct GPS coordinates), "
            "Bank Letter (dated within 6 months, full financing confirmed, official letterhead). "
            "Always encourage them to upload documents"
            "\n--- END CONTEXT ---"
        )

    proj = analysis.get("project_summary", {})
    overall = analysis.get("overall_assessment", {})
    doc_analyses = analysis.get("document_analysis", [])
    cross_val = analysis.get("cross_document_validation", [])
    action_items = analysis.get("prioritized_action_items", [])
    ai_summary = analysis.get("ai_summary", "")

    doc_lines = []
    for doc in doc_analyses:
        checklist = doc.get("checklist_results", [])
        checklist_summary = "; ".join(
            f"{c.get('criterion', '')}={c.get('risk_level', '')}"
            for c in checklist[:4]
        )
        doc_lines.append(
            f"  [{doc.get('document_type', 'Unknown')}] {doc.get('filename', '')}\n"
            f"    Status: {doc.get('status', doc.get('document_status', 'N/A'))} | "
            f"Risk: {doc.get('risk_level', 'N/A')}\n"
            f"    Decision: {str(doc.get('decision_reasoning', 'N/A'))[:150]}\n"
            f"    Recommendations: {str(doc.get('recommendations', ''))[:120]}\n"
            f"    Checklist: {checklist_summary}"
        )

    cross_lines = [
        f"  {cv.get('validation_type', '')}: {cv.get('result', '')} — {str(cv.get('details', ''))[:120]}"
        for cv in cross_val
    ]

    action_lines = [
        f"  [{item.get('priority', '')}] {item.get('action', '')}: {str(item.get('description', ''))[:120]}"
        for item in action_items[:5]
    ]

    return (
        "\n\n--- DOCUMENT ANALYSIS CONTEXT (Session-specific, latest upload) ---\n"
        f"Project      : {proj.get('project_name', 'N/A')}\n"
        f"Type         : {proj.get('project_type', 'N/A')}\n"
        f"Capacity     : {proj.get('capacity_mw', 'N/A')} MW\n"
        f"Vendor       : {analysis.get('vendor_name', 'N/A')}\n"
        f"Analysed at  : {proj.get('analysis_timestamp', 'N/A')}\n"
        f"Docs analysed: {proj.get('documents_analyzed', 0)}\n\n"
        f"Overall Assessment:\n"
        f"  Readiness Score   : {overall.get('readiness_score', 'N/A')}\n"
        f"  Risk Level        : {overall.get('risk_level', 'N/A')}\n"
        f"  Submission Status : {overall.get('submission_status', 'N/A')}\n"
        f"  Submitted / Approved / Pending / Rejected : "
        f"{overall.get('documents_submitted','N/A')} / "
        f"{overall.get('documents_approved','N/A')} / "
        f"{overall.get('documents_pending','N/A')} / "
        f"{overall.get('documents_rejected','N/A')}\n\n"
        f"AI Summary:\n  {str(ai_summary)[:500]}\n\n"
        "Document-level Analysis:\n" + ("\n".join(doc_lines) if doc_lines else "  None") + "\n\n"
        "Cross-Document Validation:\n" + ("\n".join(cross_lines) if cross_lines else "  None") + "\n\n"
        "Top Priority Actions:\n" + ("\n".join(action_lines) if action_lines else "  None")
        + "\n--- END CONTEXT ---"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/session", response_model=SessionResponse)
async def create_session():
    """
    Create a new chat session.
    Call this when the chat page opens — returns a session_id and the
    initial greeting that asks the user to upload their documents.
    """
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()

    session_doc = {
        "session_id": session_id,
        "created_at": now,
        "messages": [
            {
                "role": "assistant",
                "content": GREETING_MESSAGE,
                "timestamp": now.isoformat(),
            }
        ],
    }
    await ChatSessions.insert_one(session_doc)

    return SessionResponse(
        session_id=session_id,
        message=GREETING_MESSAGE,
        has_analysis_data=False,
    )


@router.post("/message", response_model=ChatResponse)
async def send_message(body: ChatRequest):
    """
    Send a user message and get an AI response.

    The assistant will:
    - Use the chat history stored for this session.
    - Fetch the most recent document analysis result linked to this session_id
      and inject it as context so every answer is grounded in the actual data.
    - Respond only to permit / document-validation topics.
    """
    session = await _get_session(body.session_id)
    messages_history: list = session.get("messages", [])

    # Fetch session-specific analysis (latest upload for this session)
    analysis = await _get_session_analysis(body.session_id)
    has_data = analysis is not None

    # Build system prompt + analysis context
    full_system_prompt = SYSTEM_PROMPT + _build_context_block(analysis)

    # Build OpenAI message list from stored history (cap at last 20 turns)
    openai_messages = [{"role": "system", "content": full_system_prompt}]
    for msg in messages_history[-20:]:
        openai_messages.append({"role": msg["role"], "content": msg["content"]})
    openai_messages.append({"role": "user", "content": body.message})

    try:
        response = await openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_MINI,
            messages=openai_messages,
            temperature=0.3,
            max_tokens=1024,
        )
        assistant_reply = (
            response.choices[0].message.content
            or "I'm sorry, I could not generate a response. Please try again."
        )
    except Exception as e:
        logger.error("OpenAI chat error for session %s: %s", body.session_id, e)
        raise HTTPException(status_code=500, detail="Failed to get a response from AI. Please try again.")

    now = datetime.utcnow()

    # Persist user message + assistant reply together
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
                            "content": assistant_reply,
                            "timestamp": now.isoformat(),
                        },
                    ]
                }
            }
        },
    )

    return ChatResponse(
        session_id=body.session_id,
        message=assistant_reply,
        has_analysis_data=has_data,
    )


@router.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Return full message history for a session."""
    session = await _get_session(session_id)
    created_at = session.get("created_at")
    return {
        "session_id": session_id,
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
        "messages": session.get("messages", []),
    }


@router.get("/session/{session_id}/analysis")
async def get_session_analysis(session_id: str):
    """
    Return the latest document analysis result associated with this session.
    Useful for the frontend to display the analysis panel after upload.
    """
    await _get_session(session_id)  # validates session exists
    analysis = await _get_session_analysis(session_id)
    if not analysis:
        return {"has_analysis_data": False, "data": None}
    return {"has_analysis_data": True, "data": analysis}
