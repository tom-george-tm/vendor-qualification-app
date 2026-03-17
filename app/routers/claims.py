from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from typing import List, Optional
import asyncio
import json
import base64
import logging
import datetime
import os
import random
from bson import ObjectId
from bson.errors import InvalidId
from openai import AsyncAzureOpenAI
import fitz  # PyMuPDF

from app.database import db, ResolvedClaims, ChatSessions, Claims
from app.config import settings
from app.document_analysis import build_analysis_prompt, build_aggregator_prompt
from app.schemas import DocumentAnalysisDetail
from app.workflows.claims_workflow import claim_workflow
from app.azure_blob import (
    blob_prefix_exists,
    list_blobs_in_prefix,
    list_claim_ids,
    upload_blob,
    download_blob,
)

logger = logging.getLogger(__name__)

router = APIRouter()

BLOB_PREFIX = ""  # blobs are stored as "<CLAIM_ID>/<filename>"


def generate_claim_id() -> str:
    """Generate a unique claim folder name like CLAIM_IS_765476."""
    unique_number = random.randint(100000, 999999)
    return f"CLAIM_ID_{unique_number}"


async def trigger_background_analysis(claim_id: str):
    """
    Trigger the LangGraph workflow in the background to extract details (Applicant Name, etc.)
    and update the dashboard immediately after upload.
    """
    logger.info("Triggering background analysis for claim %s", claim_id)
    initial_state = {
        "session_id": f"bg-{claim_id}",  # Background session
        "claim_id": claim_id,
        "claim_folder": f"{claim_id}/",
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
        "is_analyzed": False,
        "error": None,
    }
    try:
        await claim_workflow.ainvoke(initial_state)
        logger.info("Background analysis completed for claim %s", claim_id)
    except Exception as e:
        logger.error("Background analysis failed for claim %s: %s", claim_id, e)


@router.post("/upload-claim-documents", summary="Upload multiple PDFs for a claim")
async def upload_claim_documents(
    files: List[UploadFile] = File(
        ...,
        description="Upload document files (PDF, PNG, JPG)",
        json_schema_extra={"items": {"type": "string", "format": "binary"}},
    ),
):
    """
    Upload multiple PDF files for a claim.
    Creates a unique prefix in Azure Blob Storage (e.g. CLAIM_ID_765476/) and stores all PDFs under it.
    Returns the claim ID and list of saved blob names.
    """
    # Validate that all uploaded files are PDFs
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF. Only PDF files are allowed.",
            )

    # Generate a unique claim ID, retry if the prefix already exists in blob storage
    claim_id = generate_claim_id()
    while blob_prefix_exists(f"{claim_id}/"):
        claim_id = generate_claim_id()

    logger.info("Created claim prefix: %s/", claim_id)

    # Pre-emptively create claim in DB
    await Claims.update_one(
        {"claim_id": claim_id},
        {
            "$set": {
                "claim_id": claim_id,
                "created_at": datetime.datetime.utcnow(),
                "applicant_name": "Aravinth Kumar",
                "medical_case": "Fractured Arm",
                "hospital_name": "Apollo Hospital",
                "readiness_score": 0,
                "risk_level": "N/A",
                "submission_status": "Not Analyzed",
                "is_analyzed": False,
            }
        },
        upsert=True,
    )

    saved_files: List[str] = []
    try:
        for file in files:
            blob_name = f"{claim_id}/{file.filename}"
            contents = await file.read()
            upload_blob(blob_name, contents)
            saved_files.append(blob_name)
            logger.info("Uploaded blob: %s", blob_name)
        
        # Trigger AI analysis in the background
        await trigger_background_analysis(claim_id)
        
    except Exception as exc:
        logger.error("Failed to upload files for claim %s: %s", claim_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {exc}")

    container_url = settings.AZURE_STORAGE_CONTAINER_URL or ""
    return {
        "claim_id": claim_id,
        "folder": f"{container_url}/{claim_id}",
        "uploaded_files": saved_files,
        "total_files": len(saved_files),
    }


@router.get("/claims", summary="Get all claim folder names and details")
async def get_all_claims():
    """
    Returns a list of all claim IDs from the database (sourced from Azure folders),
    enriched with analysis details.
    """
    try:
        # Use database as primary source
        cursor = Claims.find({}).sort("created_at", -1)
        print("Fetching claims from DB",cursor)
        db_claims = [doc async for doc in cursor]
        print(f"Found {db_claims} claims in DB")
        
        # Fallback to Azure if DB is empty (helps during migration)
        if not db_claims:
            logger.info("DB Claims empty, falling back to Azure listing")
            print("DB Claims empty, falling back to Azure listing")
            claim_folders = list_claim_ids()
            db_claims = [{"claim_id": cid} for cid in claim_folders]
    except Exception as exc:
        logger.error("Failed to list claims: %s", exc)
        print(f"Failed to list claims: {exc}")
        return {"claims": [], "total": 0}

    enriched_claims = []
    for claim_doc in db_claims:
        cid = claim_doc["claim_id"]
        
        # Default details from the claim document
        details = {
            "claim_id": cid,
            "applicant_name": claim_doc.get("applicant_name", "Pending Analysis"),
            "medical_case": claim_doc.get("medical_case", "Pending Analysis"),
            "hospital_name": claim_doc.get("hospital_name", "Pending Analysis"),
            "readiness_score": claim_doc.get("readiness_score", 0),
            "risk_level": claim_doc.get("risk_level", "N/A"),
            "submission_status": claim_doc.get("submission_status", "Not Analyzed"),
            "is_analyzed": claim_doc.get("is_analyzed", False),
            "created_at": claim_doc.get("created_at")
        }

        # Check for LATEST analysis in ChatSessions (in case it was updated recently)
        session_doc = await ChatSessions.find_one({"claim_id": cid})
        
        if session_doc and session_doc.get("claim_analysis"):
            analysis = session_doc["claim_analysis"]
            proj = analysis.get("project_summary", analysis.get("claim_summary", {}))
            overall = analysis.get("overall_assessment", {})
            
            details["applicant_name"] = proj.get("applicant_name", details["applicant_name"])
            details["medical_case"] = proj.get("medical_case", details["medical_case"])
            details["hospital_name"] = proj.get("hospital_name", details["hospital_name"])
            details["readiness_score"] = overall.get("readiness_score", details["readiness_score"])
            details["risk_level"] = overall.get("risk_level", details["risk_level"])
            details["submission_status"] = overall.get("submission_status", "Analyzed")
            details["is_analyzed"] = True

        enriched_claims.append(details)

    return {"claims": enriched_claims, "total": len(enriched_claims)}


@router.post("/sync-existing-claims", summary="Sync MongoDB 'claims' collection with Azure folders")
async def sync_existing_claims():
    """
    One-time utility to ensure every folder in Azure has a record in MongoDB.
    """
    try:
        azure_ids = list_claim_ids()
        synced = 0
        for cid in azure_ids:
            # Check if exists
            exists = await Claims.find_one({"claim_id": cid})
            if not exists:
                # Create base record
                await Claims.insert_one({
                    "claim_id": cid,
                    "created_at": datetime.datetime.utcnow(),
                    "applicant_name": "Pending Analysis",
                    "medical_case": "Pending Analysis",
                    "hospital_name": "Pending Analysis",
                    "readiness_score": 0,
                    "risk_level": "N/A",
                    "submission_status": "Not Analyzed",
                    "is_analyzed": False
                })
                synced += 1
        return {"status": "success", "azure_folders": len(azure_ids), "new_claims_synced": synced}
    except Exception as exc:
        logger.error("Sync failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/claims/{claim_id}/documents", summary="List documents in a claim folder")
def get_claim_documents(claim_id: str):
    """
    Returns the list of PDFs currently present in the claim blob prefix,
    along with which of the three required document types are missing.
    """
    if not blob_prefix_exists(f"{claim_id}/"):
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")

    REQUIRED = ["Bill", "Blood report", "Medical report"]
    KEYWORDS: dict[str, list[str]] = {
        "Bill": ["bill", "invoice", "receipt", "charge", "payment"],
        "Blood report": ["blood", "cbc", "haematology", "hematology", "lab", "pathology"],
        "Medical report": ["medical", "discharge", "summary", "report", "clinical", "diagnosis", "doctor"],
    }

    blob_names = list_blobs_in_prefix(f"{claim_id}/")
    # Extract just the filename portion, keep only PDFs
    files = sorted(
        b.split("/", 1)[1]
        for b in blob_names
        if b.lower().endswith(".pdf") and "/" in b
    )

    def _detect(filename: str) -> str:
        lower = filename.lower()
        for doc_type, kws in KEYWORDS.items():
            if any(k in lower for k in kws):
                return doc_type
        return "Unknown"

    present_types = {_detect(f) for f in files}
    missing = [dt for dt in REQUIRED if dt not in present_types]

    container_url = settings.AZURE_STORAGE_CONTAINER_URL or ""
    return {
        "claim_id": claim_id,
        "folder": f"{container_url}/{claim_id}",
        "files": [
            {"filename": f, "detected_type": _detect(f)}
            for f in files
        ],
        "total_files": len(files),
        "missing_document_types": missing,
        "is_complete": len(missing) == 0,
    }


@router.post("/claims/{claim_id}/add-documents", summary="Add missing documents to an existing claim")
async def add_claim_documents(
    claim_id: str,
    files: List[UploadFile] = File(
        ...,
        description="PDF files to add to the existing claim folder",
        json_schema_extra={"items": {"type": "string", "format": "binary"}},
    ),
):
    """
    Upload additional PDF(s) into an existing claim prefix in Azure Blob Storage.

    - The claim prefix must already exist (created via POST /upload-claim-documents).
    - If a blob with the same name already exists it will be **overwritten**,
      so you can also use this endpoint to replace a defective document.
    - Returns the updated document list and which types are still missing.
    """
    if not blob_prefix_exists(f"{claim_id}/"):
        raise HTTPException(
            status_code=404,
            detail=f"Claim '{claim_id}' not found. Create it first via POST /upload-claim-documents.",
        )

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF. Only PDF files are allowed.",
            )

    REQUIRED = ["Bill", "Blood report", "Medical report"]
    KEYWORDS: dict[str, list[str]] = {
        "Bill": ["bill", "invoice", "receipt", "charge", "payment"],
        "Blood report": ["blood", "cbc", "haematology", "hematology", "lab", "pathology"],
        "Medical report": ["medical", "discharge", "summary", "report", "clinical", "diagnosis", "doctor"],
    }

    def _detect(filename: str) -> str:
        lower = filename.lower()
        for doc_type, kws in KEYWORDS.items():
            if any(k in lower for k in kws):
                return doc_type
        return "Unknown"

    added_files: List[dict] = []
    try:
        for file in files:
            blob_name = f"{claim_id}/{file.filename}"
            contents = await file.read()
            upload_blob(blob_name, contents, overwrite=True)
            added_files.append({"filename": file.filename, "detected_type": _detect(file.filename)})
            logger.info("Added blob to claim %s: %s", claim_id, blob_name)
    except Exception as exc:
        logger.error("Failed to add files to claim %s: %s", claim_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {exc}")

    # Recalculate completeness after upload
    blob_names = list_blobs_in_prefix(f"{claim_id}/")
    all_files = sorted(
        b.split("/", 1)[1]
        for b in blob_names
        if b.lower().endswith(".pdf") and "/" in b
    )
    present_types = {_detect(f) for f in all_files}
    missing = [dt for dt in REQUIRED if dt not in present_types]

    container_url = settings.AZURE_STORAGE_CONTAINER_URL or ""
    return {
        "claim_id": claim_id,
        "folder": f"{container_url}/{claim_id}",
        "added_files": added_files,
        "all_files": [
            {"filename": f, "detected_type": _detect(f)}
            for f in all_files
        ],
        "total_files": len(all_files),
        "missing_document_types": missing,
        "is_complete": len(missing) == 0,
        "message": (
            "✅ All required documents are now present. You can re-run the claim analysis."
            if not missing
            else f"⚠️ Still missing: {', '.join(missing)}. Upload them to complete the claim."
        ),
    }


def _serialize(doc: dict) -> dict:
    """Recursively convert a MongoDB document to a JSON-serialisable dict."""
    result = {}
    for k, v in doc.items():
        if k == "_id":
            result[k] = str(v)
        elif isinstance(v, datetime.datetime):
            result[k] = v.isoformat()
        elif isinstance(v, dict):
            result[k] = _serialize(v)
        elif isinstance(v, list):
            result[k] = [_serialize(i) if isinstance(i, dict) else i for i in v]
        else:
            result[k] = v
    return result


@router.get("/resolved-claims", summary="List all resolved (processed) claims")
async def get_resolved_claims(
    skip: int = 0,
    limit: int = 50,
):
    """
    Returns all claims that have been approved and processed through the chat.
    Results are sorted by most-recently resolved first.
    Use `skip` and `limit` for pagination.
    """
    cursor = ResolvedClaims.find(
        {},
        {"claim_analysis": 0},  # exclude heavy nested analysis by default
    ).sort("resolved_at", -1).skip(skip).limit(limit)

    claims = [_serialize(doc) async for doc in cursor]
    total = await ResolvedClaims.count_documents({})

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "resolved_claims": claims,
    }


@router.get("/resolved-claims/{claim_id}", summary="Get a single resolved claim by ID")
async def get_resolved_claim(claim_id: str):
    """
    Returns full details of a resolved claim including the embedded claim analysis.
    """
    doc = await ResolvedClaims.find_one({"claim_id": claim_id})
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"No resolved claim found with ID '{claim_id}'.",
        )
    return _serialize(doc)

@router.get("/claim/{claim_id}", summary="Get a single claim by ID")
async def get_claim(claim_id: str):
    """
    Returns details of a claim from the Claims collection by claim_id.
    """
    doc = await Claims.find_one({"claim_id": claim_id})
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"No claim found with ID '{claim_id}'.",
        )
    return _serialize(doc)