from fastapi import APIRouter, HTTPException, Form  
from pydantic import BaseModel
from typing import Optional
from openai import AsyncAzureOpenAI
from datetime import datetime
import uuid
import re
import json
import logging

import os

from app.database import Results, ChatSessions, ResolvedClaims, Claims
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

# ---------------------------------------------------------------------------
# System prompts — split by context for consistency & reduced hallucination
# ---------------------------------------------------------------------------

# Used by /message endpoint (document-upload chat, general Q&A)
GENERAL_SYSTEM_PROMPT = """You are an insurance case analysis agent working for an insurance company.

Your role is to:
- Welcome users and guide them through insurance case analysis and claim submission
- Answer questions about insurance policies, claim requirements, and what each document must contain
- Explain claim analysis results including checklist findings, deviations, and risk levels
- Guide clients on which documents are required (Bill, Blood Report, Medical Report) and why
- Provide actionable recommendations based on claim analysis results when available

--- REQUIRED DOCUMENTS ---

Every claim requires these 3 documents:
1. **Hospital Bill** — Itemised bill with patient details, diagnosis, treatment charges, payment summary. Must be dated within 3 months.
2. **Blood Report** — CBC, LFT, RFT or relevant tests. Must include lab name/accreditation. Dated within 1 month.
3. **Medical Report** — Diagnosis, treatment details, surgeon/physician notes. Signed by a licensed doctor. Dated within 1 month.

--- CLAIM WORKFLOW OVERVIEW ---

The claim process follows these possible outcomes:
- **Approval (Happy Path):** All documents present, checklists pass, treatment is covered. Claim is approved and settlement is processed.
- **Partial Rejection:** Some items are covered, some are excluded (e.g. cosmetic procedures under policy exclusions). Covered portion is approved; excluded portion is rejected with clear reasoning.
- **Full Rejection:** Treatment/procedure is not covered under the policy (e.g. elective/cosmetic procedures excluded by policy clause). Claim is rejected with policy clause reference and appeal rights.
- **More Info Needed:** Documents are missing or have critical deficiencies. Provider is contacted for additional information.

These are illustrative categories only — always base your responses on the ACTUAL analysis data when available.

--- END WORKFLOW OVERVIEW ---

When NO document analysis data is available yet:
- Respond naturally and helpfully to greetings, general questions, and process questions
- Guide the user toward uploading their 3 required documents
- Explain what each document should contain, common checklist criteria, and what makes a document pass or fail
- Never refuse a greeting or a reasonable question about the qualification process

When document analysis data IS available (provided in the context block below):
- Ground every answer in the actual analysis data — do NOT invent or assume any information
- Report the exact risk level, readiness score, and submission status from the analysis
- Reference the project summary, overall assessment, document-level analysis, cross-document validation, and prioritized action items as needed
- Use cross-document validation findings and action items to give precise recommendations

Only redirect the user if they ask about something completely unrelated to insurance (e.g. cooking, sports). In that case respond:
"I can only assist with insurance case analysis and document validation topics."
Tone: professional, precise, and helpful."""


# Used by /claim_message Q&A follow-ups and _get_ai_qa_response
CLAIM_GROUNDED_PROMPT = """You are an insurance claim analysis assistant. Your responses must be STRICTLY grounded in the claim analysis data provided below.

CRITICAL RULES — follow these exactly:
1. Answer ONLY using information from the provided claim analysis data. Do NOT invent, assume, or hallucinate any details.
2. When reporting risk levels, readiness scores, coverage status, or recommendations, use the EXACT values from the analysis data fields.
3. Do NOT make up risk levels, decisions, amounts, or policy clauses that are not explicitly stated in the data.
4. When asked about the recommendation or decision:
   - Report the `final_suggestion` field value (APPROVE / REJECT / MORE_INFO_NEEDED) exactly as stated
   - Report the `coverage_status` field value exactly as stated
   - Report the `risk_level` field value exactly as stated
   - Do NOT override or reinterpret these values
5. When discussing issues, list ONLY the issues from `all_detected_issues` — do not add your own.
6. If information is not present in the analysis data, say "This information is not available in the current analysis."
7. Be concise and factual. Avoid speculative language like "might", "could be", "possibly".

Tone: professional, precise, and data-driven."""


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


class StartClaimSessionRequest(BaseModel):
    claim_id: str


class ClaimSummary(BaseModel):
    claim_id: str
    applicant_name: str
    policy_number: str
    medical_case: str
    hospital_name: str
    hospital_location: str
    claimed_amount: Optional[float]
    readiness_score: float
    risk_level: str
    submission_status: str
    is_analyzed: bool
    final_suggestion: Optional[str] = None
    coverage_status: Optional[str] = None
    created_at: Optional[str] = None


class ClaimsListResponse(BaseModel):
    total: int
    claims: list


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
    full_system_prompt = GENERAL_SYSTEM_PROMPT + _build_context_block(analysis)

    # Build OpenAI message list from stored history (cap at last 20 turns)
    openai_messages = [{"role": "system", "content": full_system_prompt}]
    for msg in messages_history[-20:]:
        openai_messages.append({"role": msg["role"], "content": msg["content"]})
    openai_messages.append({"role": "user", "content": body.message})

    try:
        response = await openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_MINI,
            messages=openai_messages,
            temperature=0.2,
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
    """
    session = await _get_session(body.session_id)
    messages_history: list = session.get("messages", [])
    stored_analysis: Optional[dict] = session.get("claim_analysis")
    settlement_pending: bool = session.get("settlement_pending", False)
    provider_request_suggested: bool = session.get("provider_request_suggested", False)
    settlement_info: dict = session.get("settlement_info", {})
    now = datetime.utcnow()

    response_msg = ""
    has_data = False

    # Pre-compute reusable conditions
    _is_yes = body.message.strip().lower() == "yes"
    _analysis_not_ready = (
        stored_analysis is not None
        and stored_analysis.get("overall_assessment", {}).get("submission_status") == "Not Ready"
    )

    # ── Branch A: Settlement pending ────────────────────────────────────────
    # Exclude "REJECT" so it falls through to Branch B.5 even when settlement is pending
    if settlement_pending and settlement_info and body.message.strip().upper() != "REJECT":
        user_msg = body.message.strip().lower()
        proceed_keywords = {"yes", "proceed", "approve", "accept", "confirm", "ok", "okay"}
        is_proceed = bool(proceed_keywords & set(user_msg.split()))

        # Try to parse a custom numeric amount (increase or decrease)
        custom_amount: Optional[float] = None
        if not is_proceed:
            amt_match = re.search(r"(\d[\d,]*\.?\d*)", user_msg.replace(" ", ""))
            if amt_match:
                try:
                    custom_amount = float(amt_match.group(1).replace(",", ""))
                except ValueError:
                    custom_amount = None

        if is_proceed:
            s_claim_id = settlement_info.get("claim_id", "")
            s_amount = settlement_info.get("settlement_amount", 0)
            b_amount = settlement_info.get("bill_amount", 0)
            try:
                container = _get_container_client()
                for blob_name in list_blobs_in_prefix(f"{s_claim_id}/"):
                    container.delete_blob(blob_name)
                logger.info("Deleted blobs for claim %s after approval.", s_claim_id)
            except Exception as _e:
                logger.warning("Could not delete blobs for claim %s: %s", s_claim_id, _e)

            # Update ResolvedClaims collection
            await ResolvedClaims.update_one(
                {"claim_id": s_claim_id},
                {
                    "$set": {
                        "claim_id": s_claim_id,
                        "session_id": body.session_id,
                        "resolved_at": now,
                        "resolution_type": "approved",
                        "bill_amount": b_amount,
                        "deduction_percentage": settlement_info.get("deduction_percentage"),
                        "settlement_amount": s_amount,
                        "final_amount": s_amount,
                        "claim_analysis": stored_analysis or {},
                    }
                },
                upsert=True,
            )

            # Delete from active Claims — full record is archived in ResolvedClaims above
            from app.database import Claims
            await Claims.delete_one({"claim_id": s_claim_id})
            logger.info("Deleted claim %s from Claims collection after approval.", s_claim_id)

            # Send Emails
            applicant_name = "Customer"
            policy_number = "[Policy Number]"
            diagnosis = "Medical Condition"
            if stored_analysis:
                proj = stored_analysis.get("claim_summary", stored_analysis.get("project_summary", {}))
                applicant_name = proj.get("applicant_name", "Customer")
                policy_number = proj.get("policy_number", "[Policy Number]")
                diagnosis = proj.get("diagnosis", "Medical Condition")

            try:
                from app.email import Email
                email_sender = Email(name=applicant_name, url="")
                # Send to policyholder (uses hospital/total/approved logic internally or just reasoning if we format it)
                reasoning = (
                    f"Claim Details:\n"
                    f"Total Claimed Amount: ₹{b_amount:,.2f}\n"
                    f"Approved Amount: ₹{s_amount:,.2f}"
                )
                if b_amount > s_amount:
                    reasoning += f"\nNon-payable Amount: ₹{(b_amount - s_amount):,.2f} (non-medical expenses/co-pay as per policy terms)"
                
                # We updated the approve_email template to accept hospital_name, diagnosis, total_amount, approved_amount, non_payable_amount
                # We need to use sendMail directly to pass these new kwargs since send_approval_email doesn't accept them in the signature.
                # Actually, let's just use sendMail directly for the policyholder
                hospital_name = proj.get("hospital_name", "the hospital") if stored_analysis else "the hospital"
                
                await email_sender.sendMail(
                    subject=f'Claim Approval Notification – Policy {policy_number}',
                    template_name='approve_email',
                    claim_id=s_claim_id,
                    applicant_name=applicant_name,
                    policy_number=policy_number,
                    hospital_name=hospital_name,
                    diagnosis=diagnosis,
                    total_amount=b_amount,
                    approved_amount=s_amount,
                    non_payable_amount=(b_amount - s_amount)
                )
                
                # Send to provider
                await email_sender.send_provider_intimation_email(
                    claim_id=s_claim_id,
                    applicant_name=applicant_name,
                    total_amount=b_amount,
                    approved_amount=s_amount,
                    provider_name="Healthcare Provider"
                )
                logger.info("Sent approval and intimation emails for claim %s", s_claim_id)
            except Exception as e:
                logger.error("Error sending approval emails for %s: %s", s_claim_id, e)

            # Format Response Message (EOB Style)
            non_payable = b_amount - s_amount
            non_payable_str = f"| Non-payable Amount | **₹{non_payable:,.2f}** |\n" if non_payable > 0 else ""
            
            response_msg = (
                f"✅ **Claim `{s_claim_id}` — Approved & Processed**\n\n"
                f"🌟 *Straight-through processing case—minimal human effort, maximum efficiency.*\n\n"
                f"📊 **Coverage Breakdown (EOB):**\n"
                f"| Item | Amount |\n|------|--------|\n"
                f"| Total Claimed Amount | ₹{b_amount:,.2f} |\n"
                f"{non_payable_str}"
                f"| **Final Payable Amount** | **₹{s_amount:,.2f}** |\n\n"
                f"📤 **Automated Actions Completed:**\n"
                f"  - Claim status updated to **Approved**\n"
                f"  - EOB Generated and sent to Policyholder\n"
                f"  - Approval notification dispatched to Healthcare Provider\n\n"
                f"📧 **Emails Sent Successfully:**\n"
                f"  - ✉️ Approval notice → **{applicant_name}** (Policyholder)\n"
                f"  - ✉️ Settlement intimation → **Healthcare Provider**\n\n"
                f"---\n\n"
                f"✅ This claim is now **closed**. Ready for the next one!\n\n"
                f"👉 Please send a new **Claim ID** (e.g. `CLAIM_ID_123456`) to begin the next analysis."
            )

            await ChatSessions.update_one(
                {"session_id": body.session_id},
                {"$set": {"settlement_pending": False, "claim_analysis": None}},
            )
            has_data = True

        elif custom_amount is not None and custom_amount > 0:
            s_claim_id = settlement_info.get("claim_id", "")
            b_amount = settlement_info.get("bill_amount", 0)
            
            # Calculate if it's increase or decrease
            original_amount = settlement_info.get("settlement_amount", 0)
            change_type = "increased" if custom_amount > original_amount else "decreased"
            change_amount = abs(custom_amount - original_amount)
            
            await ResolvedClaims.update_one(
                {"claim_id": s_claim_id},
                {
                    "$set": {
                        "claim_id": s_claim_id,
                        "session_id": body.session_id,
                        "resolved_at": now,
                        "resolution_type": "custom_amount",
                        "bill_amount": b_amount,
                        "original_settlement": original_amount,
                        "final_amount": custom_amount,
                        "change_type": change_type,
                        "change_amount": change_amount,
                        "claim_analysis": stored_analysis or {},
                    }
                },
                upsert=True,
            )
            
            # Delete from active Claims
            from app.database import Claims
            await Claims.delete_one({"claim_id": s_claim_id})
            logger.info("Deleted claim %s from Claims collection after custom amount processing.", s_claim_id)
            
            # Send emails for custom amount
            applicant_name = "Customer"
            if stored_analysis:
                proj = stored_analysis.get("claim_summary", stored_analysis.get("project_summary", {}))
                applicant_name = proj.get("applicant_name", "Customer")
            
            try:
                from app.email import Email
                email_sender = Email(name=applicant_name, url="")
                
                await email_sender.sendMail(
                    subject=f'Claim Settlement Update – Custom Amount – Policy {proj.get("policy_number", "N/A")}',
                    template_name='approve_email',  # Reuse template with custom messaging
                    claim_id=s_claim_id,
                    applicant_name=applicant_name,
                    policy_number=proj.get("policy_number", "N/A"),
                    hospital_name=proj.get("hospital_name", "the hospital"),
                    diagnosis=proj.get("diagnosis", "Medical Condition"),
                    total_amount=b_amount,
                    approved_amount=custom_amount,
                    non_payable_amount=(b_amount - custom_amount)
                )
                
                await email_sender.send_provider_intimation_email(
                    claim_id=s_claim_id,
                    applicant_name=applicant_name,
                    total_amount=b_amount,
                    approved_amount=custom_amount,
                    provider_name="Healthcare Provider"
                )
                logger.info("Sent custom amount emails for claim %s", s_claim_id)
            except Exception as e:
                logger.error("Error sending custom amount emails for %s: %s", s_claim_id, e)
            
            if custom_amount < original_amount:
                negotiation_note = (
                    f"\n\n🤝 **Negotiation Accepted:** The CSR has approved a reduced settlement of **₹{custom_amount:,.2f}** "
                    f"(₹{change_amount:,.2f} below the calculated amount), reflecting the identified billing discrepancies "
                    f"such as excess room rent, non-payable consumables, and pharmacy overcharging."
                )
            else:
                negotiation_note = (
                    f"\n\n📈 **Amount Adjusted Upward:** Settlement increased by ₹{change_amount:,.2f} above the calculated figure."
                )

            response_msg = (
                f"✅ **Claim `{s_claim_id}` — Negotiated Settlement Processed**\n\n"
                f"💰 **Settlement Breakdown:**\n"
                f"| Item | Amount |\n|------|--------|\n"
                f"| Total Bill Amount | ₹{b_amount:,.2f} |\n"
                f"| Calculated Settlement | ₹{original_amount:,.2f} |\n"
                f"| **Final Approved Amount** | **₹{custom_amount:,.2f}** |\n"
                f"| {'Negotiated Deduction' if custom_amount < original_amount else 'Upward Adjustment'} | ₹{change_amount:,.2f} |"
                f"{negotiation_note}\n\n"
                f"📤 **Actions Completed:**\n"
                f"  - Settlement amount finalised at **₹{custom_amount:,.2f}**\n"
                f"  - Claim status updated to **Approved (Negotiated)**\n"
                f"  - EOB generated with negotiated amount\n"
                f"  - Notification emails sent to Policyholder & Provider\n\n"
                f"---\n\n"
                f"✅ This claim is now **closed**. Ready for the next one!\n\n"
                f"👉 Please send a new **Claim ID** to continue processing."
            )
            
            await ChatSessions.update_one(
                {"session_id": body.session_id},
                {"$set": {"settlement_pending": False, "claim_analysis": None}},
            )
            has_data = True

        else:
            # Settlement pending but user asked a question instead
            response_msg = await _get_ai_qa_response(body.message, messages_history, stored_analysis, is_settlement_pending=True)
            has_data = True

    # ── Branch B: Provider request suggested ────────────────────────────────
    # Also catches "yes" when stored_analysis indicates NOT_READY (flag may not survive session recreation)
    elif _is_yes and (provider_request_suggested or _analysis_not_ready):
        print("Provider request suggested")
        claim_id = session.get("claim_id") or (stored_analysis.get("claim_summary", {}).get("claim_id") if stored_analysis else None)
        if claim_id:
            missing_docs = []
            if stored_analysis:
                issues = stored_analysis.get("overall_assessment", {}).get("all_detected_issues", [])
                missing_docs = [i for i in issues if any(k in i.lower() for k in ["missing", "not found"])]
            
            if not missing_docs:
                missing_docs = ["Required claim documentation (Bill, Medical Report, or Blood Report)"]

            applicant_name = "Customer"
            if stored_analysis:
                applicant_name = stored_analysis.get("claim_summary", {}).get("applicant_name", "Customer")

            try:
                from app.email import Email
                email_sender = Email(name=applicant_name, url="")
                await email_sender.send_missing_info_email(claim_id=claim_id, missing_documents=missing_docs)
                
                from app.database import Claims
                await Claims.update_one({"claim_id": claim_id}, {"$set": {"submission_status": "Pending External Info"}})
                
                await ChatSessions.update_one({"session_id": body.session_id}, {"$set": {"provider_request_suggested": False}})

                response_msg = (
                    f"✅ **Action Taken:** I have sent a contextual email to the healthcare provider requesting:\n"
                    + "\n".join([f"- {d}" for d in missing_docs]) +
                    f"\n\nThe claim status for `{claim_id}` is now **'Pending External Info'**."
                )
                has_data = True
            except Exception as e:
                logger.error("Provider request error: %s", e)
                response_msg = "I encountered an error contacting the provider. Please try again."
        else:
            response_msg = "I couldn't identify the claim. Please provide a Claim ID."

    # ── Branch B.5: CSR types REJECT ────────────────────────────────────────
    # Also fires when settlement_pending=True but CSR overrides with a manual REJECT
    elif body.message.strip().upper() == "REJECT" and stored_analysis:
        print("REJECTED========",body.message)
        claim_id = session.get("claim_id", "")
        overall = stored_analysis.get("overall_assessment", {})
        proj = stored_analysis.get("claim_summary", stored_analysis.get("project_summary", {}))

        applicant_name = proj.get("applicant_name", "Customer")
        policy_number = proj.get("policy_number", "[Policy Number]")
        diagnosis = proj.get("diagnosis", proj.get("medical_case", ""))
        claimed_amount = proj.get("claimed_amount")

        # Extract the primary rejection reason (policy clause / exclusion)
        all_issues = overall.get("all_detected_issues", [])
        coverage_status = overall.get("coverage_status", "")
        ai_summary_obj = stored_analysis.get("ai_summary", {})
        ai_summary_text = ai_summary_obj.get("summary_text", "") if isinstance(ai_summary_obj, dict) else str(ai_summary_obj)

        # Try to find an exclusion-related issue as the primary rejection reason
        exclusion_keywords = ["exclusion", "not covered", "excluded", "policy clause", "cosmetic", "pre-existing", "waiting period", "non-payable"]
        exclusion_issues = [i for i in all_issues if any(k in i.lower() for k in exclusion_keywords)]
        rejection_reason = exclusion_issues[0] if exclusion_issues else (all_issues[0] if all_issues else coverage_status or "The procedure/treatment is not covered under the policy terms.")

        # Build a readable policy clause highlight
        policy_clause = ""
        clause_details = ""
        if exclusion_issues:
            policy_clause = "Policy Exclusion Applies"
            clause_details = "; ".join(exclusion_issues[:3])
        elif all_issues:
            policy_clause = "Policy Rule Violation"
            clause_details = "; ".join(all_issues[:3])

        try:
            # 0. Delete blobs for rejected claim
            try:
                container = _get_container_client()
                for blob_name in list_blobs_in_prefix(f"{claim_id}/"):
                    container.delete_blob(blob_name)
                logger.info("Deleted blobs for claim %s after rejection.", claim_id)
            except Exception as _blob_err:
                logger.warning("Could not delete blobs for claim %s: %s", claim_id, _blob_err)

            # 1. Update ResolvedClaims
            await ResolvedClaims.update_one(
                {"claim_id": claim_id},
                {
                    "$set": {
                        "claim_id": claim_id,
                        "session_id": body.session_id,
                        "resolved_at": now,
                        "resolution_type": "rejected",
                        "rejection_reason": rejection_reason,
                        "policy_clause": policy_clause,
                        "clause_details": clause_details,
                        "claim_analysis": stored_analysis,
                    }
                },
                upsert=True,
            )

            # 2. Delete from active Claims — full record is archived in ResolvedClaims above
            from app.database import Claims
            await Claims.delete_one({"claim_id": claim_id})
            logger.info("Deleted claim %s from Claims collection after rejection.", claim_id)

            # 3. Send rejection emails
            from app.email import Email
            email_sender = Email(name=applicant_name, url="")

            await email_sender.send_rejection_email(
                claim_id=claim_id,
                applicant_name=applicant_name,
                policy_number=policy_number,
                reasoning=rejection_reason,
                rejection_reason=rejection_reason,
                policy_clause=policy_clause,
                clause_details=clause_details,
                diagnosis=diagnosis,
            )

            await email_sender.send_provider_rejection_email(
                claim_id=claim_id,
                applicant_name=applicant_name,
                rejection_reason=rejection_reason,
                policy_clause=policy_clause,
                clause_details=clause_details,
                diagnosis=diagnosis,
                claimed_amount=claimed_amount,
                provider_name="Healthcare Provider",
            )
            logger.info("Sent rejection emails (policyholder + provider) for claim %s", claim_id)
        except Exception as e:
            logger.error("Error processing rejection for claim %s: %s", claim_id, e)

        # 4. Build rich response
        issues_md = "\n".join([f"  • {i}" for i in all_issues[:5]]) if all_issues else "  • " + rejection_reason
        clause_md = f"\n\n📋 **Policy Clause / Exclusion Applied:**\n  `{policy_clause}`" if policy_clause else ""
        if clause_details:
            clause_md += f"\n  {clause_details}"

        response_msg = (
            f"🚫 **Claim `{claim_id}` — Rejected**\n\n"
            f"*Rejection is transparent, traceable, and aligned with policy wording—reducing disputes.*\n\n"
            f"---\n\n"
            f"**🔍 AI Insight:** Procedure not covered OR policy exclusion applies\n\n"
            f"**⚠️ Reason for Rejection:**\n{issues_md}"
            f"{clause_md}\n\n"
            f"---\n\n"
            f"📤 **Automated Actions Completed:**\n"
            f"  - Claim status updated to **Rejected**\n"
            f"  - Compliant rejection notice sent to **Policyholder** ({applicant_name})\n"
            f"  - Rejection notification dispatched to **Healthcare Provider**\n\n"
            f"📧 **Emails Sent Successfully:**\n"
            f"  - ✉️ Rejection notice → **{applicant_name}** (Policyholder) — includes policy clause & appeal rights\n"
            f"  - ✉️ Rejection notification → **Healthcare Provider**\n\n"
            f"> The policyholder has been informed of their right to **appeal within 30 days**.\n\n"
            f"---\n\n"
            f"✅ This claim is now **closed**. Ready for the next one!\n\n"
            f"👉 Please send a new **Claim ID** (e.g. `CLAIM_ID_123456`) to begin the next analysis."
        )

        # Clear session state
        await ChatSessions.update_one(
            {"session_id": body.session_id},
            {"$set": {"settlement_pending": False, "claim_analysis": None, "provider_request_suggested": False}},
        )
        has_data = True

    # ── Branch C: Normal follow-up Q&A ──────────────────────────────────────
    elif stored_analysis:
        response_msg = await _get_ai_qa_response(body.message, messages_history, stored_analysis)
        has_data = True

    # ── Branch D: First message (Claim ID or Claims Selection) ──────────────────────────────────
    else:
        # Check if this is a claim selection session
        available_claim_ids = session.get("available_claim_ids", [])
        all_claims_data = session.get("all_claims_data", [])
        processing_mode = session.get("processing_mode")
        
        if processing_mode == "claims_selection" and available_claim_ids:
            # Handle claims selection from database processing
            user_msg = body.message.strip().lower()
            
            # Check for filtering requests
            if any(keyword in user_msg for keyword in ["high risk", "high-risk", "high"]):
                # Filter high-risk claims
                high_risk_claims = [claim for claim in all_claims_data if claim.get("risk_level") == "High"]
                if high_risk_claims:
                    claims_text = "\n\n".join([
                        f"• **{claim['claim_id']}** - {claim['applicant_name']} - ₹{claim.get('claimed_amount', 0):,} - 🔴 High Risk"
                        for claim in high_risk_claims
                    ])
                    response_msg = f"🔴 **High-Risk Claims Found:**\n\n{claims_text}\n\nWhich high-risk claim would you like to process?"
                else:
                    response_msg = "✅ No high-risk claims found. All claims are low or medium risk."
            
            elif any(keyword in user_msg for keyword in ["medium risk", "medium-risk", "medium"]):
                # Filter medium-risk claims
                medium_risk_claims = [claim for claim in all_claims_data if claim.get("risk_level") == "Medium"]
                if medium_risk_claims:
                    claims_text = "\n\n".join([
                        f"• **{claim['claim_id']}** - {claim['applicant_name']} - ₹{claim.get('claimed_amount', 0):,} - 🟡 Medium Risk"
                        for claim in medium_risk_claims
                    ])
                    response_msg = f"🟡 **Medium-Risk Claims Found:**\n\n{claims_text}\n\nWhich medium-risk claim would you like to process?"
                else:
                    response_msg = "✅ No medium-risk claims found."
            
            elif any(keyword in user_msg for keyword in ["low risk", "low-risk", "low"]):
                # Filter low-risk claims
                low_risk_claims = [claim for claim in all_claims_data if claim.get("risk_level") == "Low"]
                if low_risk_claims:
                    claims_text = "\n\n".join([
                        f"• **{claim['claim_id']}** - {claim['applicant_name']} - ₹{claim.get('claimed_amount', 0):,} - 🟢 Low Risk"
                        for claim in low_risk_claims
                    ])
                    response_msg = f"🟢 **Low-Risk Claims Found:**\n\n{claims_text}\n\nWhich low-risk claim would you like to process?"
                else:
                    response_msg = "✅ No low-risk claims found."
            
            elif any(keyword in user_msg for keyword in ["analyzed", "ready", "processed"]):
                # Filter analyzed claims
                analyzed_claims = [claim for claim in all_claims_data if claim.get("is_analyzed")]
                if analyzed_claims:
                    claims_text = "\n\n".join([
                        f"• **{claim['claim_id']}** - {claim['applicant_name']} - ₹{claim.get('claimed_amount', 0):,} - ✅ Analyzed"
                        for claim in analyzed_claims
                    ])
                    response_msg = f"✅ **Analyzed Claims Ready:**\n\n{claims_text}\n\nWhich analyzed claim would you like to process?"
                else:
                    response_msg = "⏳ No analyzed claims found yet. Please wait for analysis to complete."
            
            elif any(keyword in user_msg for keyword in ["pending", "not analyzed", "waiting"]):
                # Filter pending claims
                pending_claims = [claim for claim in all_claims_data if not claim.get("is_analyzed")]
                if pending_claims:
                    claims_text = "\n\n".join([
                        f"• **{claim['claim_id']}** - {claim['applicant_name']} - ₹{claim.get('claimed_amount', 0):,} - ⏳ Pending Analysis"
                        for claim in pending_claims
                    ])
                    response_msg = f"⏳ **Claims Pending Analysis:**\n\n{claims_text}\n\nWhich claim would you like to analyze first?"
                else:
                    response_msg = "✅ All claims have been analyzed."
            
            elif any(keyword in user_msg for keyword in ["show all", "list all", "all claims", "claims"]):
                # Show all claims
                claims_text = "\n\n".join([
                    f"• **{claim['claim_id']}** - {claim['applicant_name']} - ₹{claim.get('claimed_amount', 0):,} - {claim.get('risk_level', 'N/A')} Risk"
                    for claim in all_claims_data
                ])
                response_msg = f"📋 **All Available Claims:**\n\n{claims_text}\n\nWhich claim would you like to process?"
            
            else:
                # Check if user specified a claim ID from the available list
                matching_claim = None
                for claim in all_claims_data:
                    if claim["claim_id"].lower() in user_msg:
                        matching_claim = claim
                        break
                
                if matching_claim:
                    # Process the specified claim
                    claim_id = matching_claim["claim_id"]
                    initial_state = {
                        "session_id": body.session_id, "claim_id": claim_id, "claim_folder": "",
                        "raw_files": [], "classified_files": [], "missing_docs": [],
                        "document_analyses": [], "aggregated_result": {}, "bill_amount": None,
                        "settlement_amount": None, "deduction_percentage": 0, "is_ready": False,
                        "response_message": "", "error": None,
                    }
                    result = await claim_workflow.ainvoke(initial_state)
                    response_msg = result["response_message"]
                    agg = result.get("aggregated_result", {})
                    has_data = bool(agg)
                    
                    # Add context about available claims
                    available_claims_text = "\n\n**🔄 Available Claims:**\n" + "\n".join([f"• {claim['claim_id']}" for claim in all_claims_data])
                    response_msg += f"\n\n{available_claims_text}\n\n💡 **Tip:** You can switch claims anytime by mentioning another Claim ID, or ask to filter by risk level!"

                    if has_data:
                        is_ready = result.get("is_ready", False)
                        settle_amt = result.get("settlement_amount")
                        s_pending = bool(is_ready and settle_amt is not None)
                        
                        overall = agg.get("overall_assessment", {})
                        is_not_ready = overall.get("submission_status") == "Not Ready"
                        suggestion = overall.get("final_suggestion")
                        detected_issues = overall.get("all_detected_issues", [])
                        coverage_status = overall.get("coverage_status", "")

                        # Check for policy exclusion
                        exclusion_keywords = ["exclusion", "not covered", "excluded", "policy clause",
                                              "cosmetic", "pre-existing", "waiting period", "non-payable"]
                        has_policy_exclusion = any(
                            any(k in issue.lower() for k in exclusion_keywords)
                            for issue in detected_issues
                        )
                        is_not_covered = "not covered" in coverage_status.lower() if coverage_status else False

                        # Only suggest provider request for missing-docs scenarios,
                        # NOT for policy exclusion/coverage rejection (those go to Branch B.5 REJECT).
                        if has_policy_exclusion or is_not_covered:
                            suggest_provider = False   # Scenario 3 — CSR will type REJECT
                        else:
                            suggest_provider = (suggestion == "REJECT" or is_not_ready)

                        try:
                            await ChatSessions.update_one(
                                {"session_id": body.session_id},
                                {
                                    "$set": {
                                        "claim_id": claim_id,
                                        "current_claim_id": claim_id,
                                        "claim_analysis": agg,
                                        "settlement_pending": s_pending,
                                        "settlement_info": {
                                            "claim_id": claim_id,
                                            "bill_amount": result.get("bill_amount"),
                                            "settlement_amount": settle_amt,
                                            "deduction_percentage": result.get("deduction_percentage", 0),
                                        } if s_pending else {},
                                        "provider_request_suggested": suggest_provider
                                    }
                                },
                            )
                            logger.info(
                                "Session %s updated: provider_request_suggested=%s, settlement_pending=%s",
                                body.session_id, suggest_provider, s_pending,
                            )
                        except Exception as _update_err:
                            logger.error("Failed to update session state for %s: %s", body.session_id, _update_err)
                
                elif user_msg in ["hello", "hi", "help", "start", "begin"]:
                    # Show welcome with all claims
                    claims_display = []
                    for i, claim in enumerate(all_claims_data, 1):
                        status_emoji = "✅" if claim["is_analyzed"] else "⏳"
                        risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴", "N/A": "⚪"}.get(claim["risk_level"], "⚪")
                        
                        claims_display.append(
                            f"{i}. **{claim['claim_id']}** {status_emoji} {risk_emoji}\n"
                            f"   👤 {claim['applicant_name']} | 🏥 {claim['hospital_name']}\n"
                            f"   💰 ₹{claim.get('claimed_amount', 0):,} | 📊 {claim['risk_level']} Risk\n"
                            f"   📋 {claim['submission_status']}"
                        )
                    
                    response_msg = (
                        f"🎯 **Welcome to Claims Processing Center!**\n\n"
                        f"Found **{len(all_claims_data)} claims** in your database. "
                        f"Please select a claim by mentioning its **Claim ID** (e.g., `CLAIM_ID_123456`)\n\n"
                        f"**📋 Available Claims:**\n\n"
                        + "\n\n".join(claims_display) +
                        f"\n\n---\n\n"
                        f"💡 **How to use:**\n"
                        f"• Type a **Claim ID** to analyze that claim\n"
                        f"• Ask questions like \"What is the risk level of CLAIM_ID_123456?\"\n"
                        f"• Say \"Switch to CLAIM_ID_789012\" to change claims\n"
                        f"• Use **approve**, **reject**, or suggest **custom amounts** for settlements\n"
                        f"• Ask \"Show me all high-risk claims\" for filtering\n\n"
                        f"🚀 **Ready to begin!** Which claim would you like to process first?"
                    )
                
                else:
                    # Default response for claims selection mode
                    response_msg = (
                        f"I found {len(all_claims_data)} claims in database.\n\n"
                        f"**Available Actions:**\n"
                        f"• Mention a **Claim ID** to process it (e.g., `CLAIM_ID_123456`)\n"
                        f"• Ask to **filter by risk level**: \"Show high-risk claims\"\n"
                        f"• Ask for **analyzed claims**: \"Show analyzed claims\"\n"
                        f"• Ask for **pending claims**: \"Show pending claims\"\n"
                        f"• Say **show all claims** for complete list\n\n"
                        f"Which action would you like to take?"
                    )
        
        else:
            # Original claim ID processing logic
            match = re.fullmatch(r"(CLAIM_ID_\d+)", body.message.strip())
            if not match and body.message.strip().lower() != "hello":
                response_msg = (
                    "👋 **Welcome to the Claim Analysis Assistant!**\n\n"
                    "Please send your **Claim ID** (e.g. `CLAIM_ID_123456`) to begin analysis."
                )
            else:
                claim_id = match.group(1) if match else None
                if claim_id:
                    initial_state = {
                        "session_id": body.session_id, "claim_id": claim_id, "claim_folder": "",
                        "raw_files": [], "classified_files": [], "missing_docs": [],
                        "document_analyses": [], "aggregated_result": {}, "bill_amount": None,
                        "settlement_amount": None, "deduction_percentage": 0, "is_ready": False,
                        "response_message": "", "error": None,
                    }
                    result = await claim_workflow.ainvoke(initial_state)
                    response_msg = result["response_message"]
                    agg = result.get("aggregated_result", {})
                    has_data = bool(agg)

            if has_data:
                is_ready = result.get("is_ready", False)
                settle_amt = result.get("settlement_amount")
                s_pending = bool(is_ready and settle_amt is not None)
                
                overall = agg.get("overall_assessment", {})
                is_not_ready = overall.get("submission_status") == "Not Ready"
                suggestion = overall.get("final_suggestion")
                detected_issues = overall.get("all_detected_issues", [])
                coverage_status = overall.get("coverage_status", "")

                # Check for policy exclusion
                exclusion_keywords = ["exclusion", "not covered", "excluded", "policy clause",
                                      "cosmetic", "pre-existing", "waiting period", "non-payable"]
                has_policy_exclusion = any(
                    any(k in issue.lower() for k in exclusion_keywords)
                    for issue in detected_issues
                )
                is_not_covered = "not covered" in coverage_status.lower() if coverage_status else False

                # Only suggest provider request for missing-docs scenarios,
                # NOT for policy exclusion/coverage rejection (those go to Branch B.5 REJECT).
                if has_policy_exclusion or is_not_covered:
                    suggest_provider = False   # Scenario 3 — CSR will type REJECT
                else:
                    suggest_provider = (suggestion == "REJECT" or is_not_ready)

                try:
                    await ChatSessions.update_one(
                        {"session_id": body.session_id},
                        {
                            "$set": {
                                "claim_id": claim_id,
                                "claim_analysis": agg,
                                "settlement_pending": s_pending,
                                "settlement_info": {
                                    "claim_id": claim_id,
                                    "bill_amount": result.get("bill_amount"),
                                    "settlement_amount": settle_amt,
                                    "deduction_percentage": result.get("deduction_percentage", 0),
                                } if s_pending else {},
                                "provider_request_suggested": suggest_provider
                            }
                        },
                    )
                    logger.info(
                        "Session %s updated: provider_request_suggested=%s, settlement_pending=%s",
                        body.session_id, suggest_provider, s_pending,
                    )
                except Exception as _update_err:
                    logger.error("Failed to update session state for %s: %s", body.session_id, _update_err)

    # ── Finalise turns ──────────────────────────────────────────────────────
    await ChatSessions.update_one(
        {"session_id": body.session_id},
        {
            "$push": {
                "messages": {
                    "$each": [
                        {"role": "user", "content": body.message, "timestamp": now.isoformat()},
                        {"role": "assistant", "content": response_msg, "timestamp": now.isoformat()},
                    ]
                }
            }
        },
    )

    return ChatResponse(session_id=body.session_id, message=response_msg, has_analysis_data=has_data)


async def _get_ai_qa_response(message: str, history: list, analysis: dict, is_settlement_pending: bool = False) -> str:
    """Helper to get a strictly grounded Q&A response from OpenAI."""
    context_json = json.dumps(analysis, indent=2, default=str)
    system_content = CLAIM_GROUNDED_PROMPT + "\n\n"
    if is_settlement_pending:
        system_content += "A settlement offer is pending. Remind the user they can reply 'proceed' or suggest a different amount.\n\n"
    
    system_content += f"CLAIM ANALYSIS DATA:\n{context_json[:8000]}"

    openai_msgs = [{"role": "system", "content": system_content}]
    for msg in history[-10:]:
        openai_msgs.append({"role": msg["role"], "content": msg["content"]})
    openai_msgs.append({"role": "user", "content": message})

    try:
        resp = await openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_MINI,
            messages=openai_msgs,
            temperature=0.0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or "I couldn't generate a response."
    except Exception as e:
        logger.error("Q&A error: %s", e)
        return "I encountered an error while processing your question."

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


@router.post("/session/{session_id}/switch-claim", response_model=ChatResponse)
async def switch_claim_in_session(
    session_id: str,
    claim_id: str = Form(...)
):
    """
    Switch to a different claim within the same session.
    This allows natural conversation flow when processing multiple claims.
    """
    session = await _get_session(session_id)
    available_claim_ids = session.get("available_claim_ids", [])
    processing_mode = session.get("processing_mode")
    
    # Validate claim ID is available
    if processing_mode == "claim_ids" and available_claim_ids:
        if claim_id not in available_claim_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Claim '{claim_id}' is not in the available claims list"
            )
    else:
        # For regular sessions, verify claim exists in database
        from app.database import Claims
        claim_doc = await Claims.find_one({"claim_id": claim_id})
        if not claim_doc:
            raise HTTPException(
                status_code=404,
                detail=f"Claim '{claim_id}' not found"
            )
    
    # Process the new claim
    initial_state = {
        "session_id": session_id, "claim_id": claim_id, "claim_folder": "",
        "raw_files": [], "classified_files": [], "missing_docs": [],
        "document_analyses": [], "aggregated_result": {}, "bill_amount": None,
        "settlement_amount": None, "deduction_percentage": 0, "is_ready": False,
        "response_message": "", "error": None,
    }
    result = await claim_workflow.ainvoke(initial_state)
    response_msg = result["response_message"]
    agg = result.get("aggregated_result", {})
    has_data = bool(agg)
    
    # Update session with new claim context
    await ChatSessions.update_one(
        {"session_id": session_id},
        {
            "$set": {
                "claim_id": claim_id,
                "current_claim_id": claim_id,
                "claim_analysis": agg,
                "settlement_pending": False,
                "provider_request_suggested": False,
            }
        }
    )
    
    # Add context about available claims if in claim_ids mode
    if processing_mode == "claim_ids" and available_claim_ids:
        available_claims_text = "\n\n**Available Claims:**\n" + "\n".join([f"• {cid}" for cid in available_claim_ids])
        response_msg += f"\n\n{available_claims_text}\n\nYou can switch to any claim by mentioning its Claim ID."
    
    return ChatResponse(
        session_id=session_id,
        message=response_msg,
        has_analysis_data=has_data
    )


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


# ---------------------------------------------------------------------------
# Claims-first chat flow
# ---------------------------------------------------------------------------

@router.get("/claims", response_model=ClaimsListResponse)
async def list_claims_for_chat():
    """
    Returns all active claims from the database, enriched with the latest
    analysis data (readiness score, risk level, suggestion, etc.).

    Call this to populate the "pick a claim" list before creating a session.
    Resolved/deleted claims are excluded because they no longer exist in the
    Claims collection.
    """
    cursor = Claims.find({}).sort([("created_at", -1)])
    db_claims = [doc async for doc in cursor]

    if not db_claims:
        return ClaimsListResponse(total=0, claims=[])

    enriched: list = []
    for claim_doc in db_claims:
        cid = claim_doc["claim_id"]

        detail = {
            "claim_id": cid,
            "applicant_name": claim_doc.get("applicant_name", "Pending Analysis"),
            "policy_number": claim_doc.get("policy_number", "Pending Analysis"),
            "medical_case": claim_doc.get("medical_case", "Pending Analysis"),
            "diagnosis": claim_doc.get("diagnosis", "Pending Analysis"),
            "procedure": claim_doc.get("procedure", "Pending Analysis"),
            "hospital_name": claim_doc.get("hospital_name", "Pending Analysis"),
            "hospital_location": claim_doc.get("hospital_location", "Pending Analysis"),
            "claimed_amount": claim_doc.get("claimed_amount"),
            "readiness_score": claim_doc.get("readiness_score", 0),
            "risk_level": claim_doc.get("risk_level", "N/A"),
            "submission_status": claim_doc.get("submission_status", "Not Analyzed"),
            "is_analyzed": claim_doc.get("is_analyzed", False),
            "final_suggestion": None,
            "coverage_status": None,
            "created_at": (
                claim_doc["created_at"].isoformat()
                if isinstance(claim_doc.get("created_at"), datetime)
                else claim_doc.get("created_at")
            ),
        }

        # Enrich from the most recent analysis session for this claim
        session_doc = await ChatSessions.find_one(
            {"claim_id": cid}, sort=[("created_at", -1)]
        )
        if session_doc and session_doc.get("claim_analysis"):
            analysis = session_doc["claim_analysis"]
            proj = analysis.get("project_summary", analysis.get("claim_summary", {}))
            overall = analysis.get("overall_assessment", {})

            # Only enrich when the analysis has meaningful data
            if overall and proj.get("applicant_name") not in (None, "", "N/A", "Pending Analysis"):
                detail["applicant_name"] = proj.get("applicant_name", detail["applicant_name"])
                detail["policy_number"] = proj.get("policy_number", detail["policy_number"])
                detail["medical_case"] = proj.get("medical_case", detail["medical_case"])
                detail["diagnosis"] = proj.get("diagnosis", detail["diagnosis"])
                detail["procedure"] = proj.get("procedure", detail["procedure"])
                detail["hospital_name"] = proj.get("hospital_name", detail["hospital_name"])
                detail["hospital_location"] = proj.get("hospital_location", detail["hospital_location"])
                detail["claimed_amount"] = proj.get("claimed_amount", detail["claimed_amount"])
                detail["readiness_score"] = overall.get("readiness_score", detail["readiness_score"])
                detail["risk_level"] = overall.get("risk_level", detail["risk_level"])
                detail["submission_status"] = overall.get("submission_status", detail["submission_status"])
                detail["is_analyzed"] = True
                detail["final_suggestion"] = overall.get("final_suggestion")
                detail["coverage_status"] = overall.get("coverage_status")

        enriched.append(detail)

    return ClaimsListResponse(total=len(enriched), claims=enriched)


@router.post("/start-claim-session")
async def start_claim_session(body: StartClaimSessionRequest):
    """
    Creates a new chat session tied to the given claim_id and immediately
    runs the LangGraph claim workflow so the first response already contains
    the full analysis.

    Flow:
      1. Verify the claim exists in the Claims collection.
      2. Create a new ChatSession document.
      3. Run claim_workflow and persist the result on the session.
      4. Return session_id + the AI's first analysis message so the frontend
         can drop straight into the claim_message conversation loop.

    After this call, use POST /api/chat/claim_message with the returned
    session_id to continue the conversation (approve / reject / revise /
    Q&A — all handled there with full email notifications).
    """
    claim_id = body.claim_id.strip()

    # 1. Verify claim exists
    claim_doc = await Claims.find_one({"claim_id": claim_id})
    if not claim_doc:
        raise HTTPException(
            status_code=404,
            detail=f"Claim '{claim_id}' not found in database.",
        )

    # 2. Create session
    session_id = str(uuid.uuid4())
    now = datetime.utcnow()
    session_doc = {
        "session_id": session_id,
        "claim_id": claim_id,
        "current_claim_id": claim_id,
        "processing_mode": "single_claim",
        "created_at": now,
        "messages": [],
    }
    await ChatSessions.insert_one(session_doc)
    logger.info("Created session %s for claim %s", session_id, claim_id)

    # 3. Run the LangGraph workflow
    initial_state = {
        "session_id": session_id,
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

    try:
        result = await claim_workflow.ainvoke(initial_state)
    except Exception as exc:
        logger.error("Workflow failed for claim %s: %s", claim_id, exc)
        raise HTTPException(status_code=500, detail=f"Claim analysis failed: {exc}")

    response_msg: str = result.get("response_message", "")
    agg: dict = result.get("aggregated_result", {})
    has_data = bool(agg)

    # 4. Compute session flags (same logic as Branch D in claim_message)
    is_ready = result.get("is_ready", False)
    settle_amt = result.get("settlement_amount")
    s_pending = bool(is_ready and settle_amt is not None)

    suggest_provider = False
    if has_data:
        overall = agg.get("overall_assessment", {})
        is_not_ready = overall.get("submission_status") == "Not Ready"
        suggestion = overall.get("final_suggestion")
        detected_issues = overall.get("all_detected_issues", [])
        coverage_status = overall.get("coverage_status", "")

        exclusion_keywords = [
            "exclusion", "not covered", "excluded", "policy clause",
            "cosmetic", "pre-existing", "waiting period", "non-payable",
        ]
        has_policy_exclusion = any(
            any(k in issue.lower() for k in exclusion_keywords)
            for issue in detected_issues
        )
        is_not_covered = "not covered" in coverage_status.lower() if coverage_status else False

        if has_policy_exclusion or is_not_covered:
            suggest_provider = False
        else:
            suggest_provider = suggestion == "REJECT" or is_not_ready

    # 5. Persist workflow result on session
    session_update: dict = {
        "claim_analysis": agg if has_data else None,
        "settlement_pending": s_pending,
        "provider_request_suggested": suggest_provider,
        "settlement_info": (
            {
                "claim_id": claim_id,
                "bill_amount": result.get("bill_amount"),
                "settlement_amount": settle_amt,
                "deduction_percentage": result.get("deduction_percentage", 0),
            }
            if s_pending
            else {}
        ),
    }

    # Store the opening AI message in history
    if response_msg:
        session_update["messages"] = [
            {
                "role": "assistant",
                "content": response_msg,
                "timestamp": now.isoformat(),
            }
        ]

    await ChatSessions.update_one(
        {"session_id": session_id},
        {"$set": session_update},
    )
    logger.info(
        "Session %s initialised: settlement_pending=%s, provider_request_suggested=%s",
        session_id, s_pending, suggest_provider,
    )

    # 6. Build claim summary for the response
    claim_detail = {
        "claim_id": claim_id,
        "applicant_name": claim_doc.get("applicant_name", "Pending Analysis"),
        "policy_number": claim_doc.get("policy_number", "Pending Analysis"),
        "medical_case": claim_doc.get("medical_case", "Pending Analysis"),
        "hospital_name": claim_doc.get("hospital_name", "Pending Analysis"),
        "claimed_amount": claim_doc.get("claimed_amount"),
        "readiness_score": claim_doc.get("readiness_score", 0),
        "risk_level": claim_doc.get("risk_level", "N/A"),
        "submission_status": claim_doc.get("submission_status", "Not Analyzed"),
        "is_analyzed": has_data,
    }
    if has_data:
        proj = agg.get("project_summary", agg.get("claim_summary", {}))
        overall_a = agg.get("overall_assessment", {})
        claim_detail.update({
            "applicant_name": proj.get("applicant_name", claim_detail["applicant_name"]),
            "policy_number": proj.get("policy_number", claim_detail["policy_number"]),
            "medical_case": proj.get("medical_case", claim_detail["medical_case"]),
            "hospital_name": proj.get("hospital_name", claim_detail["hospital_name"]),
            "claimed_amount": proj.get("claimed_amount", claim_detail["claimed_amount"]),
            "readiness_score": overall_a.get("readiness_score", claim_detail["readiness_score"]),
            "risk_level": overall_a.get("risk_level", claim_detail["risk_level"]),
            "submission_status": overall_a.get("submission_status", claim_detail["submission_status"]),
            "final_suggestion": overall_a.get("final_suggestion"),
            "coverage_status": overall_a.get("coverage_status"),
            "settlement_pending": s_pending,
            "settlement_amount": settle_amt,
            "bill_amount": result.get("bill_amount"),
        })

    return {
        "session_id": session_id,
        "claim_id": claim_id,
        "message": response_msg,
        "has_analysis_data": has_data,
        "claim": claim_detail,
    }
