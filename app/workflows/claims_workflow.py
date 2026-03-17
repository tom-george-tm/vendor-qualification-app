"""
LangGraph workflow for insurance claim document analysis.

Nodes
-----
1. validate_claim      – Validate the claim ID format and confirm folder exists.
2. load_documents      – Read all PDFs from the claim folder into memory.
3. classify_documents  – Determine each PDF's document type (Bill / Blood report /
                         Medical report) via filename keywords + AI fallback.
4. analyze_documents   – Run the per-document OpenAI vision checklist in parallel.
5. aggregate           – Call the aggregator prompt to produce the overall assessment.
6. format_response     – Build a human-readable chat reply from the aggregated result.
"""

import asyncio
import base64
import json
import logging
import re
from typing import List, Optional

import fitz  # PyMuPDF
from langgraph.graph import END, StateGraph
from openai import AsyncAzureOpenAI
from typing_extensions import TypedDict

from app.database import ChatSessions, Claims
from app.config import settings
from app.document_analysis import build_aggregator_prompt, build_analysis_prompt
from app.azure_blob import blob_prefix_exists, list_blobs_in_prefix, download_blob
from app.email import Email

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_DOC_TYPES = ["Bill", "Blood report", "Medical report"]

SETTLEMENT_THRESHOLD = 100000  # ₹1,00,000 (1 lakh)
DEDUCTION_PERCENTAGE = 10      # 10% co-payment for amounts above threshold

# Simple keyword → document type mapping (checked against the lowercased filename)
_FILENAME_KEYWORDS: dict[str, list[str]] = {
    "Bill": ["bill", "invoice", "receipt", "charge", "payment"],
    "Blood report": ["blood", "cbc", "haematology", "hematology", "lab", "pathology", "haemo"],
    "Medical report": ["medical", "discharge", "summary", "report", "clinical", "diagnosis", "doctor", "prescription"],
}

# ---------------------------------------------------------------------------
# OpenAI client (shared within this module)
# ---------------------------------------------------------------------------

_openai_client = AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class RawFile(TypedDict):
    filename: str
    content: bytes  # raw PDF bytes – kept in-memory only, never persisted


class ClassifiedFile(TypedDict):
    filename: str
    content: bytes
    doc_type: str  # "Bill" | "Blood report" | "Medical report" | "Unknown"


class ClaimWorkflowState(TypedDict):
    # Inputs
    session_id: str
    claim_id: str

    # Populated by nodes
    claim_folder: str
    raw_files: List[RawFile]
    classified_files: List[ClassifiedFile]
    missing_docs: List[str]
    document_analyses: List[dict]
    aggregated_result: dict

    # Settlement
    bill_amount: Optional[float]
    settlement_amount: Optional[float]
    deduction_percentage: Optional[float]
    is_ready: bool

    # Output
    response_message: str
    is_analyzed: bool
    error: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_by_filename(filename: str) -> Optional[str]:
    """Return a doc type based on filename keywords, or None if ambiguous."""
    lower = filename.lower()
    for doc_type, keywords in _FILENAME_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return doc_type
    return None


async def _ai_classify(file_info: RawFile) -> Optional[str]:
    """
    Extract the first-page text of a PDF and ask the model to classify it
    as one of the three required document types.
    """
    try:
        pdf = fitz.open(stream=file_info["content"], filetype="pdf")
        first_page_text = pdf[0].get_text()[:1500]
        pdf.close()

        response = await _openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_MINI,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document classifier for insurance claims. "
                        "Classify the document as exactly one of: "
                        "'Bill', 'Blood report', 'Medical report'. "
                        "Respond with only the type label, nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Document first-page text:\n{first_page_text}",
                },
            ],
            temperature=0.0,
            max_tokens=10,
        )
        label = (response.choices[0].message.content or "").strip()
        return label if label in REQUIRED_DOC_TYPES else None
    except Exception as exc:
        logger.warning(
            "AI classification failed for '%s': %s", file_info["filename"], exc
        )
        return None


async def _analyze_single_doc(file_info: ClassifiedFile) -> dict:
    """Convert a PDF to page images and run the OpenAI vision checklist analysis."""
    doc_type = file_info["doc_type"]
    filename = file_info["filename"]
    content = file_info["content"]

    analysis_prompt = build_analysis_prompt(doc_type)
    user_content: list = [
        {"type": "text", "text": f"Analyse this {doc_type} document (file: {filename}):"}
    ]

    # Convert PDF pages → base64 PNG images
    try:
        pdf = fitz.open(stream=content, filetype="pdf")
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        pdf.close()
    except Exception as exc:
        return {
            "filename": filename,
            "document_type": doc_type,
            "error": f"Failed to render PDF: {exc}",
        }

    try:
        response = await _openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_MINI,
            messages=[
                {"role": "system", "content": analysis_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        result = {"error": f"OpenAI call failed: {exc}"}

    result.setdefault("filename", filename)
    result.setdefault("document_type", doc_type)
    return result


# ---------------------------------------------------------------------------
# Node 1 – Validate claim ID & folder
# ---------------------------------------------------------------------------


async def node_validate_claim(state: ClaimWorkflowState) -> dict:
    claim_id = (state.get("claim_id") or "").strip()

    if not re.fullmatch(r"CLAIM_ID_\d+", claim_id):
        return {
            "error": (
                f"'{claim_id}' is not a valid Claim ID. "
                "Please provide an ID in the format **CLAIM_ID_XXXXXX** (e.g. CLAIM_ID_192113)."
            )
        }

    claim_prefix = f"{claim_id}/"
    if not blob_prefix_exists(claim_prefix):
        return {
            "error": (
                f"No documents found for claim **{claim_id}** in blob storage. "
                "Please upload the claim documents first using the upload endpoint."
            )
        }

    logger.info("Claim blob prefix validated: %s", claim_prefix)
    return {"claim_folder": claim_prefix, "error": None}


# ---------------------------------------------------------------------------
# Node 2 – Load PDF files from the folder
# ---------------------------------------------------------------------------


async def node_load_documents(state: ClaimWorkflowState) -> dict:
    claim_prefix = state["claim_folder"]  # e.g. "CLAIM_ID_123456/"
    blob_names = list_blobs_in_prefix(claim_prefix)
    pdf_blob_names = sorted(b for b in blob_names if b.lower().endswith(".pdf"))

    if not pdf_blob_names:
        return {
            "error": (
                f"No PDF files found in blob storage for claim **{state['claim_id']}**. "
                "Please upload at least one PDF and try again."
            )
        }

    raw_files: List[RawFile] = []
    for blob_name in pdf_blob_names:
        filename = blob_name.split("/", 1)[1] if "/" in blob_name else blob_name
        content = download_blob(blob_name)
        raw_files.append({"filename": filename, "content": content})

    logger.info("Loaded %d PDF(s) for claim %s", len(raw_files), state["claim_id"])
    return {"raw_files": raw_files}


# ---------------------------------------------------------------------------
# Node 3 – Classify documents
# ---------------------------------------------------------------------------


async def node_classify_documents(state: ClaimWorkflowState) -> dict:
    classified: List[ClassifiedFile] = []
    needs_ai: List[RawFile] = []

    for file_info in state["raw_files"]:
        doc_type = _classify_by_filename(file_info["filename"])
        if doc_type:
            classified.append({**file_info, "doc_type": doc_type})  # type: ignore[misc]
        else:
            needs_ai.append(file_info)

    # AI fallback for unrecognised filenames
    if needs_ai:
        ai_types = await asyncio.gather(*[_ai_classify(f) for f in needs_ai])
        for file_info, doc_type in zip(needs_ai, ai_types):
            classified.append(
                {**file_info, "doc_type": doc_type or "Unknown"}  # type: ignore[misc]
            )

    found_types = {c["doc_type"] for c in classified}
    missing_docs = [dt for dt in REQUIRED_DOC_TYPES if dt not in found_types]

    logger.info(
        "Claim %s – classified %d doc(s), missing: %s",
        state["claim_id"],
        len(classified),
        missing_docs or "none",
    )
    return {"classified_files": classified, "missing_docs": missing_docs}


# ---------------------------------------------------------------------------
# Node 4 – Analyse each document via OpenAI
# ---------------------------------------------------------------------------


async def node_analyze_documents(state: ClaimWorkflowState) -> dict:
    tasks = [_analyze_single_doc(f) for f in state["classified_files"]]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    document_analyses: List[dict] = []
    for i, res in enumerate(raw_results):
        if isinstance(res, Exception):
            logger.error(
                "Analysis failed for '%s': %s",
                state["classified_files"][i]["filename"],
                res,
            )
            document_analyses.append(
                {
                    "filename": state["classified_files"][i]["filename"],
                    "document_type": state["classified_files"][i]["doc_type"],
                    "error": str(res),
                }
            )
        else:
            document_analyses.append(res)  # type: ignore[arg-type]

    return {"document_analyses": document_analyses}


# ---------------------------------------------------------------------------
# Node 5 – Aggregate all document results
# ---------------------------------------------------------------------------


async def node_aggregate(state: ClaimWorkflowState) -> dict:
    aggregator_prompt = build_aggregator_prompt()
    input_data = json.dumps(state["document_analyses"], indent=2, default=str)

    try:
        response = await _openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_GPT4O,
            messages=[
                {"role": "system", "content": aggregator_prompt},
                {
                    "role": "user",
                    "content": f"Aggregate the following document analyses:\n\n{input_data}",
                },
            ],
            temperature=0.1,
            max_tokens=8192,
            response_format={"type": "json_object"},
        )
        aggregated: dict = json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        logger.error("Aggregator call failed: %s", exc)
        aggregated = {"error": f"Aggregator failed: {exc}"}

    # Enrich with claim-level metadata
    aggregated["document_analysis"] = state["document_analyses"]
    aggregated["claim_id"] = state["claim_id"]
    aggregated["missing_docs"] = state["missing_docs"]

    return {"aggregated_result": aggregated}


# ---------------------------------------------------------------------------
# Node 6 – Extract total bill amount from Bill documents
# ---------------------------------------------------------------------------


async def node_extract_bill_amount(state: ClaimWorkflowState) -> dict:
    """Extract the total bill amount from Bill-type documents using AI vision."""
    agg = state.get("aggregated_result", {})
    overall = agg.get("overall_assessment", {})
    submission_status = overall.get("submission_status", "Not Ready")

    if submission_status != "Ready":
        logger.info("Claim %s not ready — skipping bill amount extraction.", state["claim_id"])
        return {"bill_amount": None, "is_ready": False}

    # Find Bill documents from the classified files
    bill_files = [f for f in state.get("classified_files", []) if f["doc_type"] == "Bill"]
    if not bill_files:
        logger.warning("Claim %s is ready but no Bill document found.", state["claim_id"])
        return {"bill_amount": None, "is_ready": True}

    bill = bill_files[0]

    # Convert PDF pages to images and ask AI to extract the total amount
    try:
        pdf = fitz.open(stream=bill["content"], filetype="pdf")
        user_content: list = [
            {
                "type": "text",
                "text": (
                    "Extract the TOTAL / GRAND TOTAL bill amount from this hospital or "
                    "medical bill. Return ONLY valid JSON with the key 'total_amount' as "
                    "a plain number (no currency symbol, no commas). "
                    'Example: {"total_amount": 150000}'
                ),
            }
        ]
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        pdf.close()

        response = await _openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_MINI,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document data extractor specialising in hospital "
                        "and medical bills. Extract the total/grand-total amount and "
                        "return ONLY valid JSON."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content or "{}")
        amount = float(result.get("total_amount", 0))
        logger.info("Extracted bill amount for %s: %.2f", state["claim_id"], amount)
        return {"bill_amount": amount, "is_ready": True}
    except Exception as exc:
        logger.error("Bill amount extraction failed for %s: %s", state["claim_id"], exc)
        return {"bill_amount": None, "is_ready": True}


# ---------------------------------------------------------------------------
# Node 7 – Calculate settlement (apply co-payment deduction if needed)
# ---------------------------------------------------------------------------


async def node_calculate_settlement(state: ClaimWorkflowState) -> dict:
    """Apply co-payment deduction when the bill exceeds the threshold."""
    bill_amount = state.get("bill_amount")
    is_ready = state.get("is_ready", False)

    if not is_ready or not bill_amount:
        return {"settlement_amount": None, "deduction_percentage": 0}

    if bill_amount > SETTLEMENT_THRESHOLD:
        deduction = bill_amount * DEDUCTION_PERCENTAGE / 100
        settlement = bill_amount - deduction
        logger.info(
            "Claim %s: bill %.2f exceeds %d — %d%% deduction -> settlement %.2f",
            state["claim_id"], bill_amount, SETTLEMENT_THRESHOLD,
            DEDUCTION_PERCENTAGE, settlement,
        )
        return {
            "settlement_amount": settlement,
            "deduction_percentage": DEDUCTION_PERCENTAGE,
        }

    logger.info(
        "Claim %s: bill %.2f within threshold — no deduction applied.",
        state["claim_id"], bill_amount,
    )
    return {"settlement_amount": bill_amount, "deduction_percentage": 0}


# ---------------------------------------------------------------------------
# Node 8 – Format the final chat response
# ---------------------------------------------------------------------------


async def node_format_response(state: ClaimWorkflowState) -> dict:
    # Error path
    if state.get("error"):
        return {"response_message": f"⚠️ {state['error']}", "is_analyzed": False}

    claim_id = state["claim_id"]
    missing = state.get("missing_docs", [])
    agg = state.get("aggregated_result", {})
    overall = agg.get("overall_assessment", {})

    # Missing documents section
    missing_section = ""
    if missing:
        missing_section = (
            f"\n\n⚠️ **Missing Documents:** {', '.join(missing)}\n"
            "These documents are required for a complete claim assessment. "
            "Please upload them and resubmit."
        )

    # AI summary
    ai_summary = agg.get("ai_summary", {})
    if isinstance(ai_summary, dict):
        summary_text = ai_summary.get("summary_text", "No summary available.")
    else:
        summary_text = str(ai_summary) or "No summary available."

    # Document breakdown
    doc_lines: List[str] = []
    for doc in agg.get("document_analysis", []):
        status = doc.get("document_status", doc.get("status", "N/A"))
        risk = doc.get("risk_level", "N/A")
        doc_lines.append(
            f"  • **{doc.get('document_type', 'Unknown')}** (`{doc.get('filename', '')}`) "
            f"— Status: {status} | Risk: {risk}"
        )
    docs_section = "\n".join(doc_lines) if doc_lines else "  No documents analysed."

    # Detected Issues
    issues = overall.get("all_detected_issues", [])
    issues_section = ""
    if issues:
        issues_section = "\n\n🚨 **Identified Issues:**\n" + "\n".join([f"  • {issue}" for issue in issues])

    # Coverage & Suggestion
    coverage = overall.get("coverage_status", "Checking...")
    suggestion = overall.get("final_suggestion", "N/A")
    
    # Settlement section (only for Ready claims with an extracted bill amount)
    settlement_section = ""
    bill_amt = state.get("bill_amount")
    settle_amt = state.get("settlement_amount")
    deduction_pct = state.get("deduction_percentage", 0)

    if state.get("is_ready") and settle_amt is not None and bill_amt:
        settlement_section = "\n💰 **Settlement Details:**\n\n"
        settlement_section += "| Item | Amount |\n|------|--------|\n"
        settlement_section += f"| Total Bill Amount | ₹{bill_amt:,.2f} |\n"
        if deduction_pct and deduction_pct > 0:
            deduction_amt = bill_amt - settle_amt
            settlement_section += (
                f"| Co-payment Deduction ({deduction_pct:.0f}%) | −₹{deduction_amt:,.2f} |\n"
                f"| **Settlement Amount** | **₹{settle_amt:,.2f}** |\n\n"
                f"⚠️ The total bill amount exceeds ₹1,00,000, so a "
                f"**{deduction_pct:.0f}% co-payment deduction** has been applied.\n"
            )
        else:
            settlement_section += f"| **Settlement Amount** | **₹{settle_amt:,.2f}** |\n"

    # Call-to-action
    cta = (
        f"\n\n👉 **Suggestion: {suggestion}**\n"
        f"**Do you want to reject or approve this claim?**\n\n"
        "You can also ask me specific questions like *'What are the critical issues?'* or *'How can I fix the document gaps?'*"
    )

    response = (
        f"✅ **Claim `{claim_id}` — Analysis Complete**\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Readiness Score | **{overall.get('readiness_score', 'N/A')}/100** |\n"
        f"| Coverage Status | {coverage} |\n"
        f"| Risk Level | {overall.get('risk_level', 'N/A')} |\n"
        f"| Submission Status | {overall.get('submission_status', 'N/A')} |\n"
        f"| Critical Issues | {overall.get('critical_issues', 0)} |\n"
        f"| Moderate Issues | {overall.get('moderate_issues', 0)} |\n"
        f"| Minor Issues | {overall.get('minor_issues', 0)} |"
        f"{missing_section}"
        f"{issues_section}\n\n"
        f"**Documents Analysed:**\n{docs_section}\n\n"
        f"**Summary:**\n{summary_text}\n\n"
        f"{settlement_section}"
        f"{cta}"
    )

    return {"response_message": response, "is_analyzed": True}


# ---------------------------------------------------------------------------
# Node 9 – Send Email Notification
# ---------------------------------------------------------------------------


async def node_send_email_notification(state: ClaimWorkflowState) -> dict:
    """Send an email notification when analysis is complete."""
    if state.get("error") or not state.get("is_analyzed"):
        return {}

    agg = state.get("aggregated_result", {})
    overall = agg.get("overall_assessment", {})
    proj = agg.get("claim_summary", agg.get("project_summary", {}))
    
    claim_id = state["claim_id"]
    decision = overall.get("final_suggestion", "MORE_INFO_NEEDED").upper()
    reasoning = overall.get("coverage_status", "") + "\n" + "\n".join(overall.get("all_detected_issues", []))
    applicant_name = proj.get("applicant_name", "Customer")
    
    # If the AI extraction doesn't provide a policy number, default it to a placeholder
    policy_number = proj.get("policy_number", "[Policy Number]")

    try:
        email_sender = Email(name=applicant_name, url="")
        
        if "APPROVE" in decision:
            await email_sender.send_approval_email(
                claim_id=claim_id,
                applicant_name=applicant_name,
                policy_number=policy_number,
                reasoning=reasoning[:1000]
            )
            logger.info("Sent approval email notification for claim %s", claim_id)
        elif "REJECT" in decision:
            await email_sender.send_rejection_email(
                claim_id=claim_id,
                applicant_name=applicant_name,
                policy_number=policy_number,
                reasoning=reasoning[:1000]
            )
            logger.info("Sent rejection email notification for claim %s", claim_id)
        else:
            logger.info("Decision is %s. No approval/rejection email sent for claim %s", decision, claim_id)
            
    except Exception as e:
        logger.warning("Failed to send email notification for claim %s: %s", claim_id, e)

    return {}


# ---------------------------------------------------------------------------
# Node 10 – Sync Claims Metadata
# ---------------------------------------------------------------------------


async def node_sync_claims_metadata(state: ClaimWorkflowState) -> dict:
    """Update the central Claims collection with the latest analysis data."""
    if state.get("error") or not state.get("is_analyzed"):
        return {}

    agg = state.get("aggregated_result", {})
    overall = agg.get("overall_assessment", {})
    proj = agg.get("project_summary", agg.get("claim_summary", {}))
    claim_id = state["claim_id"]

    try:
        # 1. Update central Claims collection (primary for dashboard)
        await Claims.update_one(
            {"claim_id": claim_id},
            {
                "$set": {
                    "applicant_name": proj.get("applicant_name", "N/A"),
                    "medical_case": proj.get("medical_case", "N/A"),
                    "hospital_name": proj.get("hospital_name", "N/A"),
                    "readiness_score": overall.get("readiness_score", 0),
                    "risk_level": overall.get("risk_level", "N/A"),
                    "submission_status": overall.get("submission_status", "Analyzed"),
                    "is_analyzed": True,
                }
            }
        )
        logger.info("Synced analysis metadata to Claims collection for %s", claim_id)

        # 2. Update ChatSessions (primary for chat grounding)
        await ChatSessions.update_one(
            {"claim_id": claim_id},
            {
                "$set": {
                    "claim_analysis": agg,
                    "is_analyzed": True
                }
            },
            upsert=True # Create a default session if one doesn't exist for background analysis
        )
        logger.info("Synced analysis metadata to ChatSessions for %s", claim_id)

    except Exception as e:
        logger.warning("Failed to sync analysis metadata for %s: %s", claim_id, e)

    return {}


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------


def _route(state: ClaimWorkflowState) -> str:
    """Route to 'format_response' immediately if an error has been set."""
    return "error" if state.get("error") else "continue"


# ---------------------------------------------------------------------------
# Build & compile the graph
# ---------------------------------------------------------------------------


def build_claim_workflow():
    graph = StateGraph(ClaimWorkflowState)

    graph.add_node("validate_claim", node_validate_claim)
    graph.add_node("load_documents", node_load_documents)
    graph.add_node("classify_documents", node_classify_documents)
    graph.add_node("analyze_documents", node_analyze_documents)
    graph.add_node("aggregate", node_aggregate)
    graph.add_node("extract_bill_amount", node_extract_bill_amount)
    graph.add_node("calculate_settlement", node_calculate_settlement)
    graph.add_node("format_response", node_format_response)
    graph.add_node("send_email", node_send_email_notification)
    graph.add_node("sync_metadata", node_sync_claims_metadata)

    graph.set_entry_point("validate_claim")

    graph.add_conditional_edges(
        "validate_claim",
        _route,
        {"continue": "load_documents", "error": "format_response"},
    )
    graph.add_conditional_edges(
        "load_documents",
        _route,
        {"continue": "classify_documents", "error": "format_response"},
    )
    graph.add_edge("classify_documents", "analyze_documents")
    graph.add_edge("analyze_documents", "aggregate")
    graph.add_edge("aggregate", "extract_bill_amount")
    graph.add_edge("extract_bill_amount", "calculate_settlement")
    graph.add_edge("calculate_settlement", "format_response")
    graph.add_edge("format_response", "send_email")
    graph.add_edge("send_email", "sync_metadata")
    graph.add_edge("sync_metadata", END)

    return graph.compile()


# Compiled graph – imported and used by chat.py
claim_workflow = build_claim_workflow()
