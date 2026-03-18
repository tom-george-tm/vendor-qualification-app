from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from openai import AsyncAzureOpenAI
from datetime import datetime
import uuid
import re
import json
import logging

import os

from app.database import Results, ChatSessions, ResolvedClaims
from app.config import settings
from app.workflows.claims_workflow import claim_workflow, REQUIRED_DOC_TYPES
from app.azure_blob import list_blobs_in_prefix, _get_container_client

logger = logging.getLogger(__name__)

router = APIRouter()
openai_client = AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

REQUIRED_DOCUMENTS = [
    "Hospital Bill",
    "Blood Report",
    "Medical Report"
]

GREETING_MESSAGE = (
    "Hello! Welcome to the Insurance Case Analysis Assistant.\n\n"
)

SYSTEM_PROMPT = """You are an insurance case analysis agent working for an insurance company.

Your role is to:
- Welcome users and guide them through insurance case analysis and claim submission
- Answer questions about insurance policies, claim requirements, and what each document must contain
- Explain claim analysis results including checklist findings, deviations, and risk levels
- Guide clients on which documents are required (bill , blood report , medical report ) and why
- Provide actionable recommendations based on claim analysis results when available
- Explain APPROVE/REJECT decisions and what the client must do to address deficiencies

--- CLAIM WORKFLOW & SCENARIO MATCHING ---

Use the following two hardcoded reference scenarios to categorise and explain the user's claim.
When analysis data is available, match the claim to the closest scenario category and tailor your
responses accordingly.

**SCENARIO 1 — CLAIM SUCCESS (Happy Path)**
Policyholder    : Rahul Sharma
Policy Number   : HI-2025-INS-004782
Policy Type     : Comprehensive Health Insurance (Family Floater)
Sum Insured     : ₹10,00,000
Policy Valid    : 01-Jan-2025 to 31-Dec-2025
Hospital        : Apollo Hospitals, Chennai (Network Hospital)
Admission Date  : 15-Mar-2025
Discharge Date  : 18-Mar-2025
Diagnosis       : Acute Appendicitis
Treatment       : Laparoscopic Appendectomy
Claim Type      : Cashless
Claim ID        : CLM-2025-03-00912

Documents Submitted:
  1. Hospital Bill        — ₹1,20,000 (room charges ₹30,000 + surgery ₹60,000 + medicines ₹18,000 + diagnostics ₹12,000)
  2. Blood Report         — CBC, LFT, RFT — all within normal range, WBC elevated (confirms infection)
  3. Medical Report       — Surgeon's notes confirming acute appendicitis, laparoscopic procedure performed, no complications

Coverage Verification:
  - Policy Status         : Active
  - Treatment Covered     : YES (Appendectomy is covered under surgical benefits)
  - Sub-limits            : Room rent capped at ₹10,000/day (3 days = ₹30,000 — within limit)
  - Waiting Period        : Not applicable (acute condition)
  - Exclusions            : None triggered
  - Result                : FULLY ELIGIBLE

Claim Amount Calculation:
  - Total Hospital Bill   : ₹1,20,000
  - Eligible Amount       : ₹1,20,000 (all charges within policy limits)
  - Deductible            : ₹0 (no deductible on this plan)
  - Co-payment (10%)      : ₹12,000
  - Network Discount      : ₹5,000 (negotiated rate)
  - Final Payable Amount  : ₹1,03,000
  - Paid To               : Apollo Hospitals (cashless settlement)

Decision: APPROVED — Full claim approved.
Settlement: Payment of ₹1,03,000 processed directly to hospital.
EOB: Generated and sent to Rahul via email and SMS.
Status: CLAIM CLOSED. Records archived for compliance.

**SCENARIO 2 — CLAIM FAILURE / PARTIAL REJECTION (Negative Path)**
Policyholder    : Anita Nair
Policy Number   : HI-2024-INS-003156
Policy Type     : Standard Health Insurance (Individual)
Sum Insured     : ₹5,00,000
Policy Valid    : 15-Jun-2024 to 14-Jun-2025
Hospital        : Fortis Hospital, Mumbai (Network Hospital)
Admission Date  : 10-Mar-2025
Discharge Date  : 12-Mar-2025
Diagnosis       : Deviated Nasal Septum + Rhinoplasty (cosmetic)
Treatment       : Septoplasty (medical) + Rhinoplasty (cosmetic)
Claim Type      : Reimbursement
Claim ID        : CLM-2025-03-00987

Documents Submitted:
  1. Hospital Bill        — ₹80,000 (septoplasty ₹35,000 + rhinoplasty ₹30,000 + room ₹8,000 + medicines ₹7,000)
  2. Blood Report         — CBC, coagulation profile — all normal
  3. Medical Report       — ENT surgeon's report: septoplasty for breathing difficulty (medically necessary) + rhinoplasty for cosmetic reshaping (elective, not medically necessary)

Coverage Verification:
  - Policy Status         : Active
  - Septoplasty Covered   : YES (medically necessary surgical procedure)
  - Rhinoplasty Covered   : NO — falls under EXCLUSION CLAUSE 4.12 (cosmetic/aesthetic procedures)
  - Sub-limits            : Room rent capped at ₹5,000/day (2 days = ₹10,000 — ₹8,000 within limit)
  - Result                : PARTIALLY ELIGIBLE

Claim Amount Calculation:
  - Total Hospital Bill          : ₹80,000
  - Eligible Expenses            : ₹50,000 (septoplasty ₹35,000 + room ₹8,000 + medicines ₹7,000)
  - Non-Eligible / Excluded      : ₹30,000 (rhinoplasty — cosmetic, excluded under policy)
  - Deductible                   : ₹0
  - Co-payment (20%)             : ₹10,000 (on eligible ₹50,000)
  - Final Payable Amount         : ₹40,000
  - Rejected Amount              : ₹30,000
  - Rejection Reason             : Rhinoplasty is a cosmetic procedure excluded under Policy Exclusion Clause 4.12

Decision: PARTIALLY APPROVED.
  - Approved: ₹40,000 for septoplasty and associated medical expenses.
  - Rejected: ₹30,000 for rhinoplasty (cosmetic — policy exclusion).
EOB: Generated with clear breakdown of approved vs rejected amounts and reasons.
Settlement: ₹40,000 reimbursement processed to Anita's bank account.
Status: CLAIM CLOSED with partial rejection. Anita notified of her right to appeal within 30 days.

--- HOW TO USE THESE SCENARIOS ---
- When explaining the claim workflow to a user, reference the appropriate scenario with its hardcoded values.
- If the user's claim has full coverage and no exclusions, follow Scenario 1 (Happy Path) as a template.
- If the user's claim has exclusions, partial coverage, or rejected items, follow Scenario 2 (Negative Path) as a template.
- Always explain which parts are approved, which are rejected, and why, using the same detailed breakdown format.
- Provide clear next steps based on which scenario the claim most closely matches.
- When no analysis data is available, you can use these scenarios as illustrative examples to explain the process.

--- END CLAIM WORKFLOW ---

When NO document analysis data is available yet:
- Respond naturally and helpfully to greetings, general questions, and process questions
- Guide the user toward uploading their 3 required documents (bill, blood report, medical report)
- Explain what each document should contain, common checklist criteria, and what makes a document pass or fail
- You may reference the workflow steps above to explain the claim process
- Never refuse a greeting or a reasonable question about the qualification process

When document analysis data IS available (provided in the context block below):
- Ground every answer in the actual analysis data - do not make up any information that is not in the analysis
- Match the claim to Category A (full approval) or Category B (partial/rejection) based on the analysis results
- Reference the project summary, overall assessment, document-level analysis, cross-document validation, and prioritized action items as needed to answer the user's question
- Use cross-document validation findings and action items to give precise recommendations

Only redirect the user if they ask about something completely unrelated to insurance, claim analysis, or document validation (e.g. cooking, sports, unrelated technical topics). In that case respond:
"I can only assist with insurance case analysis and document validation topics. Please ask me about your claim submission, policy requirements, or analysis status."
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
            "Hospital Bill (itemised bill with patient details, diagnosis, treatment charges, payment summary), "
            "Blood Report (CBC, LFT, RFT or relevant tests, lab name, dated within 1 month), "
            "Medical Report (diagnosis, treatment details, surgeon/physician notes, signed by licensed doctor). "
            "Always encourage them to upload all 3 documents for a complete claim analysis."
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

    # Extract claim-specific fields from project_summary / claim_summary
    applicant = proj.get("applicant_name", analysis.get("applicant_name", "N/A"))
    medical_case = proj.get("medical_case", analysis.get("medical_case", "N/A"))
    hospital = proj.get("hospital_name", analysis.get("hospital_name", "N/A"))

    # ai_summary may be a dict with summary_text or a plain string
    if isinstance(ai_summary, dict):
        ai_summary_text = ai_summary.get("summary_text", str(ai_summary))
    else:
        ai_summary_text = str(ai_summary)

    return (
        "\n\n--- CLAIM ANALYSIS CONTEXT (Session-specific, latest upload) ---\n"
        f"Applicant    : {applicant}\n"
        f"Medical Case : {medical_case}\n"
        f"Hospital     : {hospital}\n"
        f"Submitted by : {analysis.get('vendor_name', 'N/A')}\n"
        f"Analysed at  : {proj.get('analysis_timestamp', 'N/A')}\n"
        f"Docs analysed: {proj.get('documents_analyzed', 0)}\n\n"
        f"Overall Assessment:\n"
        f"  Readiness Score   : {overall.get('readiness_score', 'N/A')}\n"
        f"  Risk Level        : {overall.get('risk_level', 'N/A')}\n"
        f"  Submission Status : {overall.get('submission_status', 'N/A')}\n"
        f"  Documents Analysed: {overall.get('documents_analyzed', 'N/A')}\n"
        f"  Critical Issues   : {overall.get('critical_issues', 0)}\n"
        f"  Moderate Issues   : {overall.get('moderate_issues', 0)}\n"
        f"  Minor Issues      : {overall.get('minor_issues', 0)}\n\n"
        f"AI Summary:\n  {ai_summary_text[:500]}\n\n"
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

@router.post("/claim_message", response_model=ChatResponse)
async def send_claim_message(body: ChatRequest):
    """
    Claim-specific chat powered by a LangGraph workflow.

    Session lifecycle
    -----------------
    1. **First message** — must be a Claim ID (``CLAIM_ID_XXXXXX``).
       The LangGraph pipeline runs automatically:
         validate → load PDFs → classify → analyse → aggregate → format reply
       The aggregated result is stored in the session for future turns.

    2. **Subsequent messages** — free-form questions answered by the AI
       using the stored claim analysis as grounding context.
    """
    session = await _get_session(body.session_id)
    messages_history: list = session.get("messages", [])
    stored_analysis: Optional[dict] = session.get("claim_analysis")
    settlement_pending: bool = session.get("settlement_pending", False)
    settlement_info: dict = session.get("settlement_info", {})
    now = datetime.utcnow()

    # ── Branch A: Settlement pending – handle proceed / custom amount ───────
    if settlement_pending and settlement_info:
        user_msg = body.message.strip()
        user_msg_lower = user_msg.lower()
        print("User message during settlement pending:", user_msg)
        proceed_keywords = {"yes", "proceed", "approve", "accept", "confirm", "ok", "okay"}
        is_proceed = bool(proceed_keywords & set(user_msg_lower.split()))
        print("Is user accepting settlement?", is_proceed)

        # Try to parse a custom numeric amount
        custom_amount: Optional[float] = None
        if not is_proceed:
            print("Checking for custom amount in user message", user_msg)
            amt_match = re.search(r"(\d[\d,]*\.?\d*)", user_msg.replace(" ", ""))
            if amt_match:
                try:
                    custom_amount = float(amt_match.group(1).replace(",", ""))
                except ValueError:
                    custom_amount = None

        if is_proceed:
            # ── Process claim at settlement amount & delete blobs ──
            s_claim_id = settlement_info.get("claim_id", "")
            s_amount = settlement_info.get("settlement_amount", 0)
            s_bill_amount = settlement_info.get("bill_amount", 0)
            s_deduction_pct = settlement_info.get("deduction_percentage", 0)
            try:
                container = _get_container_client()
                for blob_name in list_blobs_in_prefix(f"{s_claim_id}/"):
                    container.delete_blob(blob_name)
                logger.info("Deleted all blobs for claim %s after approval.", s_claim_id)
            except Exception as _e:
                logger.warning("Could not delete blobs for claim %s: %s", s_claim_id, _e)

            # Persist resolved claim to DB
            await ResolvedClaims.update_one(
                {"claim_id": s_claim_id},
                {
                    "$set": {
                        "claim_id": s_claim_id,
                        "session_id": body.session_id,
                        "resolved_at": now,
                        "resolution_type": "approved",
                        "bill_amount": s_bill_amount,
                        "deduction_percentage": s_deduction_pct,
                        "settlement_amount": s_amount,
                        "final_amount": s_amount,
                        "claim_analysis": stored_analysis or {},
                    }
                },
                upsert=True,
            )
            logger.info("Saved resolved claim %s to DB.", s_claim_id)

            response_msg = (
                f"\u2705 **Claim `{s_claim_id}` \u2014 Processed Successfully!**\n\n"
                f"\U0001F4B0 Settlement amount of **\u20b9{s_amount:,.2f}** has been approved "
                f"and processed.\n\n"
                "The claim documents have been archived and removed.\n\n"
                "Thank you for using the Claim Analysis Assistant! "
                "You can start a new claim by providing another Claim ID."
            )
            await ChatSessions.update_one(
                {"session_id": body.session_id},
                {"$set": {"settlement_pending": False, "claim_analysis": None}},
            )
            has_data = True

        elif custom_amount is not None and custom_amount > 0:
            # ── Process claim at user-specified amount & delete blobs ──
            s_claim_id = settlement_info.get("claim_id", "")
            s_bill_amount = settlement_info.get("bill_amount", 0)
            s_deduction_pct = settlement_info.get("deduction_percentage", 0)
            original_amt = settlement_info.get("settlement_amount", 0)
            try:
                container = _get_container_client()
                for blob_name in list_blobs_in_prefix(f"{s_claim_id}/"):
                    container.delete_blob(blob_name)
                logger.info("Deleted all blobs for claim %s after custom-amount approval.", s_claim_id)
            except Exception as _e:
                logger.warning("Could not delete blobs for claim %s: %s", s_claim_id, _e)

            # Persist resolved claim to DB
            await ResolvedClaims.update_one(
                {"claim_id": s_claim_id},
                {
                    "$set": {
                        "claim_id": s_claim_id,
                        "session_id": body.session_id,
                        "resolved_at": now,
                        "resolution_type": "custom_amount",
                        "bill_amount": s_bill_amount,
                        "deduction_percentage": s_deduction_pct,
                        "settlement_amount": original_amt,
                        "final_amount": custom_amount,
                        "claim_analysis": stored_analysis or {},
                    }
                },
                upsert=True,
            )
            logger.info("Saved resolved claim %s to DB (custom amount).", s_claim_id)

            response_msg = (
                f"\u2705 **Claim `{s_claim_id}` \u2014 Processed with Custom Amount!**\n\n"
                f"\U0001F4B0 Your custom claim amount of **\u20b9{custom_amount:,.2f}** has been "
                f"accepted and processed.\n"
                f"*(Original settlement amount was \u20b9{original_amt:,.2f})*\n\n"
                "The claim documents have been archived and removed.\n\n"
                "Thank you for using the Claim Analysis Assistant! "
                "You can start a new claim by providing another Claim ID."
            )
            await ChatSessions.update_one(
                {"session_id": body.session_id},
                {"$set": {"settlement_pending": False, "claim_analysis": None}},
            )
            has_data = True

        else:
            # Not a settlement action — treat as Q&A about the claim
            context_json = json.dumps(stored_analysis or {}, indent=2, default=str)
            openai_msgs = [
                {
                    "role": "system",
                    "content": (
                        "You are an insurance claim analysis assistant. "
                        "Answer the user's questions using ONLY the claim analysis data "
                        "provided below. Be precise, professional, and helpful.\n\n"
                        "IMPORTANT: A settlement offer is currently pending for this claim. "
                        "If the user seems to be asking about proceeding or the amount, "
                        "remind them they can reply 'proceed' to accept or provide a "
                        "different amount.\n\n"
                        f"CLAIM ANALYSIS DATA:\n{context_json[:8000]}"
                    ),
                }
            ]
            for msg in messages_history[-20:]:
                openai_msgs.append({"role": msg["role"], "content": msg["content"]})
            openai_msgs.append({"role": "user", "content": body.message})

            try:
                resp = await openai_client.chat.completions.create(
                    model=settings.AZURE_OPENAI_DEPLOYMENT_MINI,
                    messages=openai_msgs,
                    temperature=0.3,
                    max_tokens=1024,
                )
                response_msg = (
                    resp.choices[0].message.content
                    or "I'm sorry, I could not generate a response. Please try again."
                )
            except Exception as exc:
                logger.error("OpenAI claim Q&A error for session %s: %s", body.session_id, exc)
                raise HTTPException(status_code=500, detail="Failed to get a response from AI.")

            has_data = True

    # ── Branch B: follow-up Q&A using stored claim analysis ─────────────────
    elif stored_analysis:
        context_json = json.dumps(stored_analysis, indent=2, default=str)
        openai_msgs = [
            {
                "role": "system",
                "content": (
                    "You are an insurance claim analysis assistant. "
                    "Answer the user's questions using ONLY the claim analysis data "
                    "provided below. Be precise, professional, and helpful.\n\n"
                    f"CLAIM ANALYSIS DATA:\n{context_json[:8000]}"
                ),
            }
        ]
        for msg in messages_history[-20:]:
            openai_msgs.append({"role": msg["role"], "content": msg["content"]})
        openai_msgs.append({"role": "user", "content": body.message})

        try:
            resp = await openai_client.chat.completions.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_MINI,
                messages=openai_msgs,
                temperature=0.3,
                max_tokens=1024,
            )
            response_msg = (
                resp.choices[0].message.content
                or "I'm sorry, I could not generate a response. Please try again."
            )
        except Exception as exc:
            logger.error("OpenAI claim Q&A error for session %s: %s", body.session_id, exc)
            raise HTTPException(status_code=500, detail="Failed to get a response from AI.")

        has_data = True

    # ── Branch C: first message — expect a Claim ID ──────────────────────────
    else:
        match = re.fullmatch(r"(CLAIM_ID_\d+)", body.message.strip())

        if not match:
            response_msg = (
                "👋 **Welcome to the Claim Analysis Assistant!**\n\n"
                "To begin, please send your **Claim ID** in the format "
                "`CLAIM_ID_XXXXXX` (e.g. `CLAIM_ID_192113`).\n\n"
                f"Required documents for analysis: {', '.join(REQUIRED_DOC_TYPES)}."
            )
            has_data = False

        else:
            claim_id = match.group(1)
            logger.info("Starting LangGraph claim workflow for %s (session %s)", claim_id, body.session_id)

            initial_state = {
                "session_id": body.session_id,
                "claim_id": claim_id,
                "claim_folder": "",
                "raw_files": [],
                "classified_files": [],
                "missing_docs": [],
                "document_analyses": [],
                "aggregated_result": {},
                "bill_amount": None,
                "settlement_amount": None,
                "deduction_percentage": 0,
                "is_ready": False,
                "response_message": "",
                "error": None,
            }

            result = await claim_workflow.ainvoke(initial_state)
            response_msg = result["response_message"]
            has_data = bool(result.get("aggregated_result"))

            # Persist analysis & settlement info to the session
            if has_data:
                is_ready = result.get("is_ready", False)
                settle_amt = result.get("settlement_amount")
                s_pending = bool(is_ready and settle_amt is not None)
                s_info = {}
                if s_pending:
                    s_info = {
                        "claim_id": claim_id,
                        "bill_amount": result.get("bill_amount"),
                        "settlement_amount": settle_amt,
                        "deduction_percentage": result.get("deduction_percentage", 0),
                    }

                await ChatSessions.update_one(
                    {"session_id": body.session_id},
                    {
                        "$set": {
                            "claim_id": claim_id,
                            "claim_analysis": result["aggregated_result"],
                            "settlement_pending": s_pending,
                            "settlement_info": s_info,
                        }
                    },
                )

    # ── Persist both turns to the message history ────────────────────────────
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
                            "content": response_msg,
                            "timestamp": now.isoformat(),
                        },
                    ]
                }
            }
        },
    )

    return ChatResponse(
        session_id=body.session_id,
        message=response_msg,
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


@router.get("/claim/{claim_id}/messages")
async def get_messages_by_claim_id(claim_id: str):
    """
    Return the full message history for the session that processed the given Claim ID.
    The claim_id is stored on the session document when the LangGraph workflow runs.
    """
    session = await ChatSessions.find_one({"claim_id": claim_id})
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"No session found for claim ID '{claim_id}'.",
        )

    created_at = session.get("created_at")
    return {
        "claim_id": claim_id,
        "session_id": session.get("session_id"),
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else created_at,
        "total_messages": len(session.get("messages", [])),
        "messages": session.get("messages", []),
    }
