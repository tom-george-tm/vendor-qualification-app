from fastapi import APIRouter, UploadFile, File, HTTPException, Form
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

from app.database import db
from app.config import settings
from app.document_analysis import build_analysis_prompt, build_aggregator_prompt
from app.schemas import DocumentAnalysisDetail

logger = logging.getLogger(__name__)

router = APIRouter()


DIR = "./temp_uploads"  # Temporary directory to save uploaded files


def generate_claim_id() -> str:
    """Generate a unique claim folder name like CLAIM_IS_765476."""
    unique_number = random.randint(100000, 999999)
    return f"CLAIM_ID_{unique_number}"


@router.post("/upload-claim-documents", summary="Upload multiple PDFs for a claim")
async def upload_claim_documents(files: List[UploadFile] = File(
        ...,
        description="Upload document files (PDF, PNG, JPG)",
        json_schema_extra={"items": {"type": "string", "format": "binary"}},
    ),):
    """
    Upload multiple PDF files for a claim.
    Creates a unique folder under DIR (e.g. CLAIM_IS_765476) and stores all PDFs in it.
    Returns the claim ID and list of saved file paths.
    """
    # Validate that all uploaded files are PDFs
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF. Only PDF files are allowed.",
            )

    # Generate a unique claim folder name, retry if the folder already exists
    claim_id = generate_claim_id()
    claim_folder = os.path.join(DIR, claim_id)
    while os.path.exists(claim_folder):
        claim_id = generate_claim_id()
        claim_folder = os.path.join(DIR, claim_id)

    os.makedirs(claim_folder, exist_ok=False)
    logger.info("Created claim folder: %s", claim_folder)

    saved_files: List[str] = []
    try:
        for file in files:
            file_path = os.path.join(claim_folder, file.filename)
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
            saved_files.append(file_path)
            logger.info("Saved file: %s", file_path)
    except Exception as exc:
        logger.error("Failed to save files for claim %s: %s", claim_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to save files: {exc}")

    return {
        "claim_id": claim_id,
        "folder": claim_folder,
        "uploaded_files": saved_files,
        "total_files": len(saved_files),
    }


@router.get("/claims", summary="Get all claim folder names")
def get_all_claims():
    """
    Returns a list of all claim folder names found under DIR.
    Only folders whose names start with 'CLAIM_ID_' are included.
    """
    if not os.path.exists(DIR):
        return {"claims": [], "total": 0}

    claim_folders = [
        name
        for name in os.listdir(DIR)
        if os.path.isdir(os.path.join(DIR, name)) and name.startswith("CLAIM_ID_")
    ]

    return {"claims": claim_folders, "total": len(claim_folders)}


@router.get("/claims/{claim_id}/documents", summary="List documents in a claim folder")
def get_claim_documents(claim_id: str):
    """
    Returns the list of PDFs currently present in the claim folder,
    along with which of the three required document types are missing.
    """
    claim_folder = os.path.join(DIR, claim_id)
    if not os.path.isdir(claim_folder):
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")

    REQUIRED = ["Bill", "Blood report", "Medical report"]
    KEYWORDS: dict[str, list[str]] = {
        "Bill": ["bill", "invoice", "receipt", "charge", "payment"],
        "Blood report": ["blood", "cbc", "haematology", "hematology", "lab", "pathology"],
        "Medical report": ["medical", "discharge", "summary", "report", "clinical", "diagnosis", "doctor"],
    }

    files = sorted(f for f in os.listdir(claim_folder) if f.lower().endswith(".pdf"))

    def _detect(filename: str) -> str:
        lower = filename.lower()
        for doc_type, kws in KEYWORDS.items():
            if any(k in lower for k in kws):
                return doc_type
        return "Unknown"

    present_types = {_detect(f) for f in files}
    missing = [dt for dt in REQUIRED if dt not in present_types]

    return {
        "claim_id": claim_id,
        "folder": claim_folder,
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
    Upload additional PDF(s) into an existing claim folder.

    - The claim folder must already exist (created via POST /upload-claim-documents).
    - If a file with the same name already exists it will be **overwritten**,
      so you can also use this endpoint to replace a defective document.
    - Returns the updated document list and which types are still missing.
    """
    claim_folder = os.path.join(DIR, claim_id)
    if not os.path.isdir(claim_folder):
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
            file_path = os.path.join(claim_folder, file.filename)
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
            added_files.append({"filename": file.filename, "detected_type": _detect(file.filename)})
            logger.info("Added file to claim %s: %s", claim_id, file_path)
    except Exception as exc:
        logger.error("Failed to add files to claim %s: %s", claim_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to save files: {exc}")

    # Recalculate completeness after upload
    all_files = sorted(f for f in os.listdir(claim_folder) if f.lower().endswith(".pdf"))
    present_types = {_detect(f) for f in all_files}
    missing = [dt for dt in REQUIRED if dt not in present_types]

    return {
        "claim_id": claim_id,
        "folder": claim_folder,
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
