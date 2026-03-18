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

**SCENARIO 3 — CLAIM REJECTION (Policy Rule Violation)**
Policyholder    : Vikram Nair
Policy Number   : HI-2025-INS-007341
Policy Type     : Standard Health Insurance (Individual)
Sum Insured     : ₹5,00,000
Policy Valid    : 01-Jan-2025 to 31-Dec-2025
Hospital        : Cloudnine Hospital, Bengaluru (Network Hospital)
Admission Date  : 10-Mar-2025
Discharge Date  : 12-Mar-2025
Diagnosis       : Refractive Error (Myopia)
Treatment       : LASIK Eye Surgery (laser vision correction)
Claim Type      : Reimbursement
Claim ID        : CLM-2025-03-01254

Documents Submitted:
  1. Hospital Bill        — ₹85,000 (procedure fee + pre-op tests + medicines)
  2. Blood Report         — Routine pre-operative CBC — all within normal range
  3. Medical Report       — Ophthalmology report confirming refractive error; LASIK recommended for vision correction

Coverage Verification:
  - Policy Status         : Active
  - Treatment Covered     : NO — falls under EXCLUSION CLAUSE 6.3 (Elective/Cosmetic Procedures — includes laser vision correction / LASIK)
  - Waiting Period        : Not applicable
  - Result                : NOT ELIGIBLE

Decision: REJECTED ENTIRELY
  - Reason: LASIK eye surgery is classified as an elective / cosmetic procedure and is explicitly excluded under Policy Exclusion Clause 6.3.
  - Applicable Clause: Exclusion Clause 6.3 — Optical / Vision Correction Procedures
Policyholder notified of rejection and right to appeal within 30 days.
Healthcare provider notified via system-generated communication.
Status: CLAIM CLOSED with full rejection. Rejection is transparent, traceable, and aligned with policy wording.

--- HOW TO USE THESE SCENARIOS ---
- When explaining the claim workflow to a user, reference the appropriate scenario with its hardcoded values.
- If the user's claim has full coverage and no exclusions, follow Scenario 1 (Happy Path) as a template.
- If the user's claim has exclusions, partial coverage, or rejected items, follow Scenario 2 (Negative Path) as a template.
- If the user's claim involves a procedure not covered under the policy (policy rule violation), follow Scenario 3 (Rejection) as a template.
- In Scenario 3, always highlight the exact policy clause/exclusion, explain the rejection reason clearly, and inform the policyholder of their right to appeal.
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

    # ── Branch A: Settlement pending ────────────────────────────────────────
    if settlement_pending and settlement_info:
        user_msg = body.message.strip().lower()
        proceed_keywords = {"yes", "proceed", "approve", "accept", "confirm", "ok", "okay"}
        is_proceed = bool(proceed_keywords & set(user_msg.split()))

        # Try to parse a custom numeric amount
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

            # Update Claims collection status
            from app.database import Claims
            await Claims.update_one(
                {"claim_id": s_claim_id},
                {"$set": {"submission_status": "Approved"}}
            )

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
                f"  - Approval notification dispatched to Healthcare Provider"
            )

            await ChatSessions.update_one(
                {"session_id": body.session_id},
                {"$set": {"settlement_pending": False, "claim_analysis": None}},
            )
            has_data = True

        elif custom_amount is not None and custom_amount > 0:
            s_claim_id = settlement_info.get("claim_id", "")
            await ResolvedClaims.update_one(
                {"claim_id": s_claim_id},
                {
                    "$set": {
                        "claim_id": s_claim_id,
                        "session_id": body.session_id,
                        "resolved_at": now,
                        "resolution_type": "custom_amount",
                        "final_amount": custom_amount,
                        "claim_analysis": stored_analysis or {},
                    }
                },
                upsert=True,
            )
            response_msg = f"✅ **Claim `{s_claim_id}` — Processed with Custom Amount!**\n\n💰 Custom amount of **₹{custom_amount:,.2f}** accepted."
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
    elif provider_request_suggested and body.message.strip().lower() == "yes":
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
                    f"\n\nThe claim status for `{claim_id}` is now **'Pending External Info'**. I'll notify you when they respond."
                )
                has_data = True
            except Exception as e:
                logger.error("Provider request error: %s", e)
                response_msg = "I encountered an error contacting the provider. Please try again."
        else:
            response_msg = "I couldn't identify the claim. Please provide a Claim ID."

    # ── Branch B.5: CSR types REJECT ────────────────────────────────────────
    elif body.message.strip().upper() == "REJECT" and stored_analysis:
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

            # 2. Update Claims status to Rejected
            from app.database import Claims
            await Claims.update_one(
                {"claim_id": claim_id},
                {"$set": {"submission_status": "Rejected"}}
            )
            logger.info("Claim %s status updated to Rejected by CSR.", claim_id)

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
            f"> The policyholder has been informed of their right to **appeal within 30 days**."
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

    # ── Branch D: First message (Claim ID) ──────────────────────────────────
    else:
        match = re.fullmatch(r"(CLAIM_ID_\d+)", body.message.strip())
        if not match:
            response_msg = (
                "👋 **Welcome to the Claim Analysis Assistant!**\n\n"
                "Please send your **Claim ID** (e.g. `CLAIM_ID_123456`) to begin analysis."
            )
        else:
            claim_id = match.group(1)
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
                suggest_provider = (suggestion == "REJECT" or is_not_ready)

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
    """Helper to get a grounded Q&A response from OpenAI."""
    context_json = json.dumps(analysis, indent=2, default=str)
    system_content = (
        "You are an insurance claim analysis assistant. "
        "Answer strictly using the provided claim analysis data.\n\n"
    )
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
            temperature=0.3,
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
