from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Optional
import json
import logging
import datetime
from bson import ObjectId
from bson.errors import InvalidId
from openai import AsyncAzureOpenAI
from app.database import Results
from app.config import settings
from app.email import Email
from app.document_analysis import build_aggregator_prompt
from app.constant.file_data import project_data  # <-- pre-analyzed document data keyed by filename

logger = logging.getLogger(__name__)

router = APIRouter()

openai_client = AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

DOC_TYPE_MAP = {
    "EIA": "EIA",
    "GRID": "Grid Assessment",
    "EPC": "EPC Contract",
    "GCAA": "GCAA NOC",
    "BANK": "Bank Letter",
}


def _detect_doc_type(filename: str) -> str:
    upper = filename.upper()
    for key, value in DOC_TYPE_MAP.items():
        if key in upper:
            return value
    return "Unknown"


def lookup_document_from_constants(filename: str, doc_type: str) -> dict:
    """
    Look up pre-analyzed document data from constants.py by filename.
    Falls back to a placeholder dict if the filename is not found.
    """
    if filename in project_data:
        logger.info("Loaded pre-analyzed data for: %s", filename)
        raw = project_data[filename]

        # ✅ Flatten: pull fields out of nested "document_analysis" to top level
        inner = raw.get("document_analysis", {})
        result = {
            "document_type": inner.get("document_type", doc_type),
            "document_status": inner.get("document_status", "Unknown"),
            "risk_level": inner.get("risk_level", "Unknown"),
            "decision_reasoning": inner.get("decision_reasoning", ""),
            "recommendations": inner.get("recommendations", []),
            "checklist_results": inner.get("checklist_results", []),
        }
    else:
        logger.warning("No pre-analyzed data found for '%s'. Using fallback placeholder.", filename)
        result = {
            "document_type": doc_type,
            "document_status": "Not Found",
            "risk_level": "Unknown",
            "decision_reasoning": f"No pre-analyzed data available for file: {filename}",
            "recommendations": [],
            "checklist_results": [],
        }

    # Always stamp filename and document_type
    result["filename"] = filename
    result["document_type"] = doc_type
    return result


async def aggregate_results_with_openai(all_doc_results: List[dict]) -> dict:
    """Aggregate individual document analyses into a project-level assessment."""
    aggregator_prompt = build_aggregator_prompt()
    input_data = json.dumps(all_doc_results, indent=2)

    try:
        response = await openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_GPT4O,
            messages=[
                {"role": "system", "content": aggregator_prompt},
                {"role": "user", "content": f"Aggregate the following document analyses:\n\n{input_data}"},
            ],
            temperature=0.1,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content or "{}"
        return json.loads(result_text)
    except Exception as e:
        logger.error("Aggregator OpenAI API call failed: %s", e)
        return {"error": f"Aggregator OpenAI API call failed: {str(e)}"}


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(
        ...,
        description="Upload document files (PDF, PNG, JPG)",
        json_schema_extra={"items": {"type": "string", "format": "binary"}},
    ),
    vendor_name: str = Form(..., description="Name of the vendor"),
    session_id: Optional[str] = Form(None, description="Chat session ID to associate with this upload"),
):
    # --- Step 1: Detect doc type for each uploaded file ---
    # We still read files so FastAPI doesn't complain, but we discard the bytes
    # since analysis is served from constants.py
    file_data: List[tuple] = []
    for file in files:
        await file.read()  # consume the stream (required by FastAPI); bytes are not used
        doc_type = _detect_doc_type(file.filename)
        file_data.append((file.filename, doc_type))

    # --- Step 2: Look up pre-analyzed data from constants.py (replaces OpenAI per-doc calls) ---
    all_doc_results: List[dict] = []
    for filename, doc_type in file_data:
        result = lookup_document_from_constants(filename, doc_type)
        all_doc_results.append(result)

    # --- Step 3: Aggregate all results via OpenAI ---
    aggregated_response = await aggregate_results_with_openai(all_doc_results)

    # --- Step 4: Attach metadata ---
    if "project_summary" not in aggregated_response:
        aggregated_response["project_summary"] = {}

    proj_summ = aggregated_response["project_summary"]
    now = datetime.datetime.utcnow()
    proj_summ["analysis_timestamp"] = now.isoformat()
    proj_summ["documents_analyzed"] = len(all_doc_results)
    if (
        not proj_summ.get("project_name")
        or proj_summ.get("project_name") == "<Extract from documents or use 'Project Al Noor' if unknown>"
    ):
        proj_summ["project_name"] = "Project Al Noor"

    aggregated_response["document_analysis"] = all_doc_results
    aggregated_response["vendor_name"] = vendor_name
    aggregated_response["created_at"] = now

    if session_id:
        aggregated_response["session_id"] = session_id

    # --- Step 5: Persist to MongoDB ---
    db_result = await Results.insert_one(aggregated_response)
    parent_id = str(db_result.inserted_id)
    aggregated_response["_id"] = parent_id

    for doc in aggregated_response["document_analysis"]:
        doc["db_id"] = parent_id

    # --- Step 6: Email notification (non-blocking, best-effort) ---
    if settings.GMAIL_CLIENT_ID and settings.GMAIL_REFRESH_TOKEN and settings.GMAIL_TO:
        try:
            overall = aggregated_response.get("overall_assessment", {})
            submission_status = overall.get("submission_status", "").upper()
            readiness_score = overall.get("readiness_score", 0)

            if submission_status in ("APPROVED", "APPROVE") or readiness_score >= 80:
                decision = "APPROVE"
            elif submission_status in ("REJECTED", "REJECT") or readiness_score < 40:
                decision = "REJECT"
            else:
                decision = None

            if decision:
                project_name = proj_summ.get("project_name", vendor_name)
                reasoning = overall.get(
                    "ai_summary",
                    aggregated_response.get("ai_summary", "See full report for details."),
                )
                email_sender = Email(name=vendor_name, url="")
                await email_sender.send_workflow_notification(
                    filename=project_name,
                    decision=decision,
                    reasoning=str(reasoning)[:500],
                )
        except Exception as email_err:
            logger.warning("Email notification failed: %s", email_err)

    aggregated_response["created_at"] = now.isoformat()
    return aggregated_response


@router.get("/results")
async def get_results():
    results = []
    async for doc in Results.find().sort("created_at", -1):
        doc["_id"] = str(doc["_id"])
        if isinstance(doc.get("created_at"), datetime.datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        results.append(doc)
    return results


@router.get("/results/{result_id}")
async def get_result_by_id(result_id: str):
    try:
        object_id = ObjectId(result_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid ID format")

    doc = await Results.find_one({"_id": object_id})
    if not doc:
        raise HTTPException(status_code=404, detail=f"Result with ID '{result_id}' not found")

    doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("created_at"), datetime.datetime):
        doc["created_at"] = doc["created_at"].isoformat()

    return doc






# from fastapi import APIRouter, UploadFile, File, HTTPException, Form
# from typing import List, Optional
# import asyncio
# import json
# import base64
# import logging
# import datetime
# from bson import ObjectId
# from bson.errors import InvalidId
# from openai import AsyncAzureOpenAI
# import fitz  # PyMuPDF
# from app.database import Results
# from app.config import settings
# from app.email import Email
# from app.document_analysis import build_analysis_prompt, build_aggregator_prompt
# from app.schemas import DocumentAnalysisDetail

# logger = logging.getLogger(__name__)

# router = APIRouter()

# openai_client = AsyncAzureOpenAI(
#     api_key=settings.AZURE_OPENAI_API_KEY,
#     azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
#     api_version=settings.AZURE_OPENAI_API_VERSION,
# )

# DOC_TYPE_MAP = {
#     "EIA": "EIA",
#     "GRID": "Grid Assessment",
#     "EPC": "EPC Contract",
#     "GCAA": "GCAA NOC",
#     "BANK": "Bank Letter",
# }


# def _detect_doc_type(filename: str) -> str:
#     upper = filename.upper()
#     for key, value in DOC_TYPE_MAP.items():
#         if key in upper:
#             return value
#     return "Unknown"


# async def analyse_document_with_openai(
#     file_content: bytes,
#     filename: str,
#     content_type: str,
#     doc_type: str,
# ) -> dict:
#     """Analyse a single document via OpenAI vision. Called in parallel for each uploaded file."""
#     analysis_prompt = build_analysis_prompt(doc_type)
#     user_content = [
#         {"type": "text", "text": f"Analyse the following {doc_type} document (filename: {filename}):"}
#     ]

#     if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
#         try:
#             pdf_document = fitz.open(stream=file_content, filetype="pdf")
#             for page_num in range(len(pdf_document)):
#                 page = pdf_document.load_page(page_num)
#                 matrix = fitz.Matrix(2.0, 2.0)
#                 pix = page.get_pixmap(matrix=matrix)
#                 img_bytes = pix.tobytes("png")
#                 b64 = base64.b64encode(img_bytes).decode("utf-8")
#                 user_content.append({
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/png;base64,{b64}"},
#                 })
#             pdf_document.close()
#         except Exception as e:
#             logger.error("Failed to convert PDF %s to images: %s", filename, e)
#             return {"error": f"Failed to process PDF file: {str(e)}"}
#     else:
#         b64 = base64.b64encode(file_content).decode("utf-8")
#         mime = content_type or "application/octet-stream"
#         user_content.append({
#             "type": "image_url",
#             "image_url": {"url": f"data:{mime};base64,{b64}"},
#         })

#     try:
#         response = await openai_client.chat.completions.create(
#             model=settings.AZURE_OPENAI_DEPLOYMENT_MINI,
#             messages=[
#                 {"role": "system", "content": analysis_prompt},
#                 {"role": "user", "content": user_content},
#             ],
#             temperature=0.1,
#             max_tokens=4096,
#             response_format={"type": "json_object"},
#         )
#         result_text = response.choices[0].message.content or "{}"
#     except Exception as e:
#         logger.error("Azure OpenAI API call failed for %s: %s", filename, e)
#         return {"error": f"Azure OpenAI API call failed: {str(e)}"}

#     try:
#         result_json = json.loads(result_text)
#     except json.JSONDecodeError:
#         result_json = {"raw_result": result_text}

#     if "risk_level" in result_json:
#         try:
#             validated = DocumentAnalysisDetail(**result_json)
#             result_json = validated.model_dump()
#         except Exception as validation_err:
#             logger.warning("Validation failed for %s: %s", filename, validation_err)

#     return result_json


# async def aggregate_results_with_openai(all_doc_results: List[dict]) -> dict:
#     """Aggregate individual document analyses into a project-level assessment."""
#     aggregator_prompt = build_aggregator_prompt()
#     input_data = json.dumps(all_doc_results, indent=2)

#     try:
#         response = await openai_client.chat.completions.create(
#             model=settings.AZURE_OPENAI_DEPLOYMENT_GPT4O,
#             messages=[
#                 {"role": "system", "content": aggregator_prompt},
#                 {"role": "user", "content": f"Aggregate the following document analyses:\n\n{input_data}"},
#             ],
#             temperature=0.1,
#             max_tokens=2048,
#             response_format={"type": "json_object"},
#         )
#         result_text = response.choices[0].message.content or "{}"
#         return json.loads(result_text)
#     except Exception as e:
#         logger.error("Aggregator OpenAI API call failed: %s", e)
#         return {"error": f"Aggregator OpenAI API call failed: {str(e)}"}


# @router.post("/upload")
# async def upload_files(
#     files: List[UploadFile] = File(
#         ...,
#         description="Upload document files (PDF, PNG, JPG)",
#         json_schema_extra={"items": {"type": "string", "format": "binary"}},
#     ),
#     vendor_name: str = Form(..., description="Name of the vendor"),
#     session_id: Optional[str] = Form(None, description="Chat session ID to associate with this upload"),
# ):
#     # --- Step 1: Read all file contents sequentially (UploadFile requires async read) ---
#     file_data: List[tuple] = []
#     for file in files:
#         content = await file.read()
#         doc_type = _detect_doc_type(file.filename)
#         file_data.append((content, file.filename, file.content_type or "application/octet-stream", doc_type))

#     # --- Step 2: Analyse all documents in parallel ---
#     tasks = [
#         analyse_document_with_openai(content, filename, content_type, doc_type)
#         for content, filename, content_type, doc_type in file_data
#     ]
#     raw_results = await asyncio.gather(*tasks, return_exceptions=True)

#     # Build results list, handling any exceptions from individual tasks
#     all_doc_results: List[dict] = []
#     for i, result in enumerate(raw_results):
#         _, filename, _, doc_type = file_data[i]
#         if isinstance(result, Exception):
#             logger.error("Parallel analysis failed for %s: %s", filename, result)
#             result_json: dict = {"error": str(result)}
#         else:
#             result_json = result  # type: ignore[assignment]
#         result_json["filename"] = filename
#         result_json["document_type"] = doc_type
#         all_doc_results.append(result_json)

#     # --- Step 3: Aggregate all results ---
#     aggregated_response = await aggregate_results_with_openai(all_doc_results)

#     # --- Step 4: Attach metadata ---
#     if "project_summary" not in aggregated_response:
#         aggregated_response["project_summary"] = {}

#     proj_summ = aggregated_response["project_summary"]
#     now = datetime.datetime.utcnow()
#     proj_summ["analysis_timestamp"] = now.isoformat()
#     proj_summ["documents_analyzed"] = len(all_doc_results)
#     if (
#         not proj_summ.get("project_name")
#         or proj_summ.get("project_name") == "<Extract from documents or use 'Project Al Noor' if unknown>"
#     ):
#         proj_summ["project_name"] = "Project Al Noor"

#     aggregated_response["document_analysis"] = all_doc_results
#     aggregated_response["vendor_name"] = vendor_name
#     aggregated_response["created_at"] = now

#     # Attach session_id so chat can query results by session
#     if session_id:
#         aggregated_response["session_id"] = session_id

#     # --- Step 5: Persist to MongoDB ---
#     db_result = await Results.insert_one(aggregated_response)
#     parent_id = str(db_result.inserted_id)
#     aggregated_response["_id"] = parent_id

#     for doc in aggregated_response["document_analysis"]:
#         doc["db_id"] = parent_id

#     # --- Step 6: Email notification (non-blocking, best-effort) ---
#     if settings.GMAIL_CLIENT_ID and settings.GMAIL_REFRESH_TOKEN and settings.GMAIL_TO:
#         try:
#             overall = aggregated_response.get("overall_assessment", {})
#             submission_status = overall.get("submission_status", "").upper()
#             readiness_score = overall.get("readiness_score", 0)

#             if submission_status in ("APPROVED", "APPROVE") or readiness_score >= 80:
#                 decision = "APPROVE"
#             elif submission_status in ("REJECTED", "REJECT") or readiness_score < 40:
#                 decision = "REJECT"
#             else:
#                 decision = None

#             if decision:
#                 project_name = proj_summ.get("project_name", vendor_name)
#                 reasoning = overall.get(
#                     "ai_summary",
#                     aggregated_response.get("ai_summary", "See full report for details."),
#                 )
#                 email_sender = Email(name=vendor_name, url="")
#                 await email_sender.send_workflow_notification(
#                     filename=project_name,
#                     decision=decision,
#                     reasoning=str(reasoning)[:500],
#                 )
#         except Exception as email_err:
#             logger.warning("Email notification failed: %s", email_err)

#     aggregated_response["created_at"] = now.isoformat()
#     return aggregated_response


# @router.get("/results")
# async def get_results():
#     results = []
#     async for doc in Results.find().sort("created_at", -1):
#         doc["_id"] = str(doc["_id"])
#         if isinstance(doc.get("created_at"), datetime.datetime):
#             doc["created_at"] = doc["created_at"].isoformat()
#         results.append(doc)
#     return results


# @router.get("/results/{result_id}")
# async def get_result_by_id(result_id: str):
#     try:
#         object_id = ObjectId(result_id)
#     except InvalidId:
#         raise HTTPException(status_code=400, detail="Invalid ID format")

#     doc = await Results.find_one({"_id": object_id})
#     if not doc:
#         raise HTTPException(status_code=404, detail=f"Result with ID '{result_id}' not found")

#     doc["_id"] = str(doc["_id"])
#     if isinstance(doc.get("created_at"), datetime.datetime):
#         doc["created_at"] = doc["created_at"].isoformat()

#     return doc
