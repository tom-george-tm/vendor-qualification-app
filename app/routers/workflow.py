from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Annotated
import json
import base64
import logging
import datetime
from bson import ObjectId
from bson.errors import InvalidId
from openai import AsyncOpenAI
import io
import fitz # PyMuPDF
from app.database import Results
from app.config import settings
from app.email import Email
from app.document_analysis import build_analysis_prompt
from app.schemas import ChatResponseFormat

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialise the async OpenAI client once
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


async def analyse_document_with_openai(
    file_content: bytes,
    filename: str,
    content_type: str,
) -> dict:
    """
    Send a document to OpenAI's vision model for analysis.
    Encodes the file as base64 and sends it as an image_url,
    alongside the structured analysis prompt.
    """
    analysis_prompt = build_analysis_prompt()
    user_content = [
        {"type": "text", "text": f"Analyse the following document (filename: {filename}):"}
    ]

    # Handle PDF by converting each page to an image using PyMuPDF
    if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                # Render page to an image (pixmap)
                # zoom factor 2.0 gives higher resolution
                matrix = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=matrix)
                # Convert to PNG format bytes
                img_bytes = pix.tobytes("png")
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            pdf_document.close()
        except Exception as e:
            logger.error("Failed to convert PDF %s to images: %s", filename, e)
            return {"error": f"Failed to process PDF file: {str(e)}"}
    else:
        # Handle standard image files
        b64 = base64.b64encode(file_content).decode("utf-8")
        mime = content_type or "application/octet-stream"
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    try:
        response = await openai_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": analysis_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content or "{}"
    except Exception as e:
        logger.error("OpenAI API call failed: %s", e)
        return {"error": f"OpenAI API call failed: {str(e)}"}

    # Parse the JSON response
    try:
        result_json = json.loads(result_text)
    except json.JSONDecodeError:
        result_json = {"raw_result": result_text}

    # Validate through Pydantic schema if it looks like an analysis response
    if "decision" in result_json:
        try:
            validated = ChatResponseFormat(**result_json)
            result_json = validated.model_dump()
        except Exception as validation_err:
            logger.warning(
                "Analysis response validation failed for %s: %s",
                filename, validation_err,
            )

    return result_json


@router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(
        ..., 
        description="Upload document files (PDF, PNG, JPG)",
        json_schema_extra={"items": {"type": "string", "format": "binary"}}
    ),
    vendor_name: str = Form(..., description="Name of the vendor"),
    input_data: str = Form("", description="Optional extra input (prompt is built automatically)"),
):

    all_results = []

    for file in files:
        content = await file.read()

        try:
            result_json = await analyse_document_with_openai(
                file_content=content,
                filename=file.filename,
                content_type=file.content_type,
            )
        except Exception as e:
            logger.error("Error processing %s: %s", file.filename, e)
            result_json = {"filename": file.filename, "error": str(e)}
        finally:
            await file.seek(0)
            
        result_json["filename"] = file.filename

        # Save payload to MongoDB
        db_record = {
            "filename": file.filename,
            "vendor_name": vendor_name,
            "result": result_json,
            "created_at": datetime.datetime.utcnow(),
        }
        
        db_result = Results.insert_one(db_record)
        result_json["db_id"] = str(db_result.inserted_id)
        
        all_results.append(result_json)

    # Return exactly what the chat response format requested in an array
    return all_results


@router.get("/results")
async def get_results():
    cursor = Results.find().sort("created_at", -1)
    results = []
    for doc in cursor:
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

    doc = Results.find_one({"_id": object_id})

    if not doc:
        raise HTTPException(status_code=404, detail=f"Result with ID '{result_id}' not found")

    doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("created_at"), datetime.datetime):
        doc["created_at"] = doc["created_at"].isoformat()

    return doc