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
import uuid
from datetime import datetime
from typing import List, Optional

import fitz  # PyMuPDF
from langgraph.graph import END, StateGraph
from openai import AsyncAzureOpenAI
from typing_extensions import TypedDict

from app.database import ChatSessions, Claims
from app.config import settings
from app.document_analysis import build_aggregator_prompt, build_analysis_prompt, compute_readiness_score
from app.azure_blob import blob_prefix_exists, list_blobs_in_prefix, download_blob, get_blob_url
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
            temperature=0.0,
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
            temperature=0.0,
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

    # ── Deterministic override: enforce decision rules in code ───────────────
    # The LLM can be inconsistent; these rules are the source of truth.
    overall = aggregated.get("overall_assessment", {})
    missing_docs = state["missing_docs"]

    # Override readiness_score with the deterministic formula:
    #   Each criterion risk_level → weight (NA=1.0, minor=0.75, moderate=0.5, critical=0.0)
    #   doc_score = avg(weights),  overall = avg(doc_scores),  score = overall × 100
    computed_score = compute_readiness_score(state["document_analyses"])
    overall["readiness_score"] = computed_score
    logger.info(
        "Claim %s — deterministic readiness score computed: %.2f",
        state["claim_id"], computed_score,
    )

    score = computed_score
    critical = overall.get("critical_issues", 0)
    moderate = overall.get("moderate_issues", 0)
    suggestion = overall.get("final_suggestion", "")
    detected_issues = overall.get("all_detected_issues", [])

    exclusion_keywords = ["exclusion", "not covered", "excluded", "policy clause",
                          "cosmetic", "pre-existing", "waiting period"]
    has_policy_exclusion = any(
        any(k in issue.lower() for k in exclusion_keywords)
        for issue in detected_issues
    )

    if has_policy_exclusion:
        # Always REJECT for policy exclusions regardless of score
        overall["final_suggestion"] = "REJECT"
        overall["submission_status"] = "Not Ready"
        overall["risk_level"] = "Critical"
    elif missing_docs:
        # Always MORE_INFO_NEEDED when documents are physically missing
        overall["final_suggestion"] = "MORE_INFO_NEEDED"
        overall["submission_status"] = "Not Ready"
        if overall.get("risk_level") not in ("High", "Critical"):
            overall["risk_level"] = "High"
    elif score >= 80 and critical == 0:
        # Happy path: all docs present, score good, no critical issues → APPROVE
        overall["final_suggestion"] = "APPROVE"
        overall["submission_status"] = "Ready"
        if overall.get("risk_level") not in ("Low", "Moderate"):
            overall["risk_level"] = "Low" if moderate == 0 else "Moderate"
    elif critical >= 3 or has_policy_exclusion:
        overall["final_suggestion"] = "REJECT"
        overall["submission_status"] = "Not Ready"
    else:
        overall["final_suggestion"] = "MORE_INFO_NEEDED"
        overall["submission_status"] = "Not Ready"

    aggregated["overall_assessment"] = overall
    logger.info(
        "Claim %s — deterministic override applied: score=%s critical=%s missing=%s → %s",
        state["claim_id"], score, critical, missing_docs,
        overall["final_suggestion"],
    )
    # ── End deterministic override ───────────────────────────────────────────

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
        filename = doc.get('filename', '')
        
        # Build the blob URL if we have a filename
        doc_link = f"`{filename}`"
        if filename:
            blob_path = f"{claim_id}/{filename}"
            url = get_blob_url(blob_path)
            doc_link = f"[{filename}]({url})"

        doc_lines.append(
            f"  • **{doc.get('document_type', 'Unknown')}** ({doc_link}) "
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

    # Call-to-action — route to the correct Scenario
    detected_issues = overall.get("all_detected_issues", [])
    exclusion_keywords = ["exclusion", "not covered", "excluded", "policy clause", "cosmetic", "pre-existing", "waiting period", "non-payable"]
    has_policy_exclusion = any(
        any(k in issue.lower() for k in exclusion_keywords)
        for issue in detected_issues
    )
    coverage = overall.get("coverage_status", "")
    is_not_covered = "not covered" in coverage.lower() if coverage else False

    if suggestion == "REJECT" or overall.get("submission_status") == "Not Ready":
        # Determine whether this is a policy exclusion scenario or a missing-docs scenario
        missing_info_items = [i for i in detected_issues if any(k in i.lower() for k in ["missing", "not found", "absent", "incomplete"])]
        all_missing = list(dict.fromkeys(list(missing) + missing_info_items))

        if has_policy_exclusion or is_not_covered:
            # ── Scenario 3: Policy Rule Violation ──────────────────────────
            exclusion_issues = [i for i in detected_issues if any(k in i.lower() for k in exclusion_keywords)]
            clause_highlight = ""
            if exclusion_issues:
                clause_highlight = (
                    f"\n\n🔒 **Policy Clause / Exclusion Identified:**\n"
                    + "\n".join([f"  • {e}" for e in exclusion_issues[:3]])
                )

            ai_recommendation = (
                f"\n\n---\n\n"
                f"## 🤖 AI Recommendation: **REJECT**\n\n"
                f"> The procedure or treatment is excluded under the policy terms. "
                f"The claim does not qualify for reimbursement based on the current policy coverage.\n"
            )

            cta = (
                f"{clause_highlight}"
                f"{ai_recommendation}\n"
                f"👉 **Type `REJECT` to proceed** — I will:\n"
                f"  1. Update the claim status to **Rejected**\n"
                f"  2. Generate a clear, compliant rejection communication\n"
                f"  3. Send the rejection notice to the **Policyholder** via email\n"
                f"  4. Dispatch a **Notification to the Healthcare Provider**\n"
            )
        else:
            # ── Scenario 2: Missing documents / info needed ─────────────────
            missing_list_md = ""
            if all_missing:
                missing_list_md = "\n".join([f"  - {item}" for item in all_missing]) + "\n"

            ai_recommendation = (
                f"\n\n---\n\n"
                f"## 🤖 AI Recommendation: **REQUEST MORE INFO**\n\n"
                f"> One or more required documents are missing or incomplete. "
                f"The claim cannot be processed until the healthcare provider submits the outstanding information.\n"
            )

            cta = (
                f"{ai_recommendation}\n"
                f"⚠️ **Required information is missing.**\n"
                f"{missing_list_md}\n"
                f"👉 **Type `YES` to proceed** — I will:\n"
                f"  1. Auto-generate a contextual email listing the missing documents\n"
                f"  2. Send it to the healthcare provider via the integrated communication system\n"
                f"  3. Update the claim status to **Pending External Info**\n"
            )
    else:
        # ── Scenario 1: Approve path ────────────────────────────────────────
        score = overall.get("readiness_score", 0)
        moderate = overall.get("moderate_issues", 0)
        if moderate > 0:
            confidence_note = f"with {moderate} minor/moderate item(s) noted — review the details above before confirming"
        else:
            confidence_note = "all documents are valid and all checklist criteria are met"

        ai_recommendation = (
            f"\n\n---\n\n"
            f"## 🤖 AI Recommendation: **APPROVE**\n\n"
            f"> Readiness score **{score}/100** — {confidence_note}.\n"
        )

        cta = (
            f"{ai_recommendation}\n"
            f"👉 **Type `YES` or `PROCEED` to approve** — I will:\n"
            f"  1. Process the settlement and generate the EOB\n"
            f"  2. Send the approval notice to the **Policyholder**\n"
            f"  3. Dispatch a settlement intimation to the **Healthcare Provider**\n\n"
            f"Or type `REJECT` to override and reject this claim instead.\n\n"
            f"You can also ask me questions like *'What are the moderate issues?'* or *'Why is the score not 100?'*"
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
    """
    Send an internal analysis-complete notification to the CSR team only.

    APPROVE and REJECT emails with full details are sent AFTER CSR confirmation
    in chat.py (Branch A and Branch B.5). This node must NOT send those emails
    to avoid duplicates — it only logs the outcome for MORE_INFO_NEEDED cases.
    """
    if state.get("error") or not state.get("is_analyzed"):
        return {}

    agg = state.get("aggregated_result", {})
    overall = agg.get("overall_assessment", {})
    claim_id = state["claim_id"]
    decision = overall.get("final_suggestion", "MORE_INFO_NEEDED").upper()

    # APPROVE → email sent by Branch A (chat.py) after CSR confirms settlement
    # REJECT  → email sent by Branch B.5 (chat.py) after CSR types REJECT
    # Both include richer details (amounts, policy clauses, provider intimation)
    # so we do NOT send here to avoid duplicates.
    logger.info(
        "Node 9 email skip: claim %s decision=%s — CSR confirmation branch will handle emails.",
        claim_id, decision,
    )
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
                    "policy_number": proj.get("policy_number", "N/A"),
                    "applicant_name": proj.get("applicant_name", "N/A"),
                    "applicant_age": proj.get("applicant_age"),
                    "patient_gender": proj.get("patient_gender"),
                    "medical_case": proj.get("medical_case", "N/A"),
                    "diagnosis": proj.get("diagnosis", "N/A"),
                    "procedure": proj.get("procedure", "N/A"),
                    "hospital_name": proj.get("hospital_name", "N/A"),
                    "hospital_location": proj.get("hospital_location", "N/A"),
                    "readiness_score": overall.get("readiness_score", 0),
                    "risk_level": overall.get("risk_level", "N/A"),
                    "submission_status": overall.get("submission_status", "Analyzed"),
                    "claimed_amount": proj.get("claimed_amount"),
                    "uploaded_documents": [
                        {
                            "filename": d.get("filename"),
                            "document_type": d.get("document_type"),
                            "status": d.get("document_status")
                        }
                        for d in agg.get("document_analysis", [])
                    ],
                    "is_analyzed": True,
                }
            }
        )
        logger.info("Synced analysis metadata to Claims collection for %s", claim_id)

        # 2. Update ChatSessions (primary for chat grounding)
        # First check if a session already exists for this claim_id
        existing_session = await ChatSessions.find_one({"claim_id": claim_id})
        if existing_session:
            # Update the existing session
            await ChatSessions.update_one(
                {"claim_id": claim_id},
                {
                    "$set": {
                        "claim_analysis": agg,
                        "is_analyzed": True
                    }
                },
            )
        else:
            # No session exists yet — create one with a proper session_id
            # to avoid violating the unique index on session_id (null conflicts).
            bg_session_id = f"bg-{claim_id}-{uuid.uuid4().hex[:8]}"
            await ChatSessions.insert_one({
                "session_id": bg_session_id,
                "claim_id": claim_id,
                "claim_analysis": agg,
                "is_analyzed": True,
                "created_at": datetime.utcnow(),
                "messages": [],
            })
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
