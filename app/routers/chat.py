from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Optional, Annotated
import httpx
import json
import datetime

from app.database import Results
from app.config import settings
from app.email import Email
from app.agent import generate_chat_response

router = APIRouter()


@router.post("/message")
async def chat_message(
    messages: Annotated[str, Form(...)],
    files: Annotated[Optional[List[UploadFile]], File()] = None,
    vendor_name: Annotated[Optional[str], Form()] = None,
):
    """
    Chat endpoint:
    1. If files are present, runs them through the existing workflow API and saves to MongoDB.
    2. Passes the conversation history + workflow result to the OpenAI agent.
    3. Returns the AI-generated response.
    """
    try:
        messages_list = json.loads(messages)
        if not isinstance(messages_list, list):
            raise ValueError("messages must be a JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid messages payload: {exc}")

    workflow_result = None
    result_id = None
    filename = None

    # Process uploaded files through the existing workflow
    active_files = [f for f in (files or []) if f.filename]
    if active_files and vendor_name:
        file = active_files[0]
        filename = file.filename
        content = await file.read()

        async with httpx.AsyncClient() as http_client:
            try:
                response = await http_client.post(
                    settings.WORKFLOW_API_URL,
                    files={"files": (file.filename, content, file.content_type)},
                    data={
                        "input_data": json.dumps(
                            {"user_input": "analyse this document and provide a compliance verdict"}
                        )
                    },
                    timeout=120.0,
                )

                if response.status_code == 200:
                    resp_data = response.json()
                    result_str = resp_data.get("result", "{}")
                    try:
                        workflow_result = json.loads(result_str)
                    except json.JSONDecodeError:
                        workflow_result = {"raw_result": result_str}

                    db_record = {
                        "filename": file.filename,
                        "vendor_name": vendor_name,
                        "result": workflow_result,
                        "created_at": datetime.datetime.utcnow(),
                    }
                    db_result = Results.insert_one(db_record)
                    result_id = str(db_result.inserted_id)

                    decision = workflow_result.get("decision", "").upper()
                    if decision in ("APPROVE", "REJECT"):
                        email = Email(name="Admin", url="")
                        await email.send_workflow_notification(
                            file.filename,
                            decision,
                            workflow_result.get("decision_reasoning", ""),
                        )

            except HTTPException:
                raise
            except Exception as exc:
                # Workflow failure is non-fatal — agent will still respond
                print(f"[chat] Workflow error: {exc}")

    # Generate AI response via OpenAI
    assistant_message = await generate_chat_response(
        messages=messages_list,
        workflow_result=workflow_result,
        filename=filename,
    )

    return {
        "message": assistant_message,
        "result_id": result_id,
        "has_result": workflow_result is not None,
    }
