from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Annotated
import httpx
import json
from app.database import Results
from app.config import settings
from app.email import Email
import datetime
from bson import ObjectId
from bson.errors import InvalidId

router = APIRouter()

@router.post("/upload")
async def upload_files(files: Annotated[List[UploadFile], File(...)], vendor_name: str, input_data: str):
    results_saved = []
    
    async with httpx.AsyncClient() as client:
        for file in files:
            # Read file content
            content = await file.read()
            
            # Prepare form data
            files_to_send = {'files': (file.filename, content, file.content_type)}
            data_to_send = {
                'input_data': json.dumps({"user_input": "analysis this document and provide a response"})
            }
            
            try:
                # Send request to external workflow API
                response = await client.post(
                    settings.WORKFLOW_API_URL,
                    files=files_to_send,
                    data=data_to_send,
                    timeout=60.0 # Standard timeout might be too short for processing
                )
                
                if response.status_code != 200:
                    results_saved.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": f"API returned {response.status_code}: {response.text}"
                    })
                    continue
                
                response_data = response.json()
                
                # Extract and parse the result string
                result_str = response_data.get("result", "{}")
                try:
                    result_json = json.loads(result_str)
                except json.JSONDecodeError:
                    result_json = {"raw_result": result_str}
                
                # Prepare data for MongoDB
                db_record = {
                    "filename": file.filename,
                    "vendor_name": vendor_name,
                    "result": result_json,
                    "created_at": datetime.datetime.utcnow()
                }
                
                # Save to MongoDB
                db_result = Results.insert_one(db_record)
                db_record["_id"] = str(db_result.inserted_id)
                
                # Check for decision and send mail
                decision = result_json.get("decision", "").upper()
                if decision in ["APPROVE", "REJECT"]:
                    email = Email(
                        name="Admin", # Generic name
                        url="" # No specific URL needed
                    )
                    await email.send_workflow_notification(file.filename, decision, result_json.get("decision_reasoning", ""))
                
                results_saved.append({
                    "filename": file.filename,
                    "status": "success",
                    "db_id": str(db_result.inserted_id)
                })
                
            except Exception as e:
                results_saved.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
            finally:
                await file.seek(0) # Reset file pointer if needed for other operations
                
    return {"status": "completed", "details": results_saved}

@router.get("/results")
async def get_results():
    cursor = Results.find().sort("created_at", -1)
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        # Ensure created_at is string if it's a datetime object
        if isinstance(doc.get("created_at"), datetime.datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        results.append(doc)
    return results


@router.get("/results/{result_id}")
async def get_result_by_id(result_id: str):
    # Validate the ObjectId format
    try:
        object_id = ObjectId(result_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    
    # Query MongoDB by _id
    doc = Results.find_one({"_id": object_id})
    
    if not doc:
        raise HTTPException(status_code=404, detail=f"Result with ID '{result_id}' not found")
    
    # Serialize the document
    doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("created_at"), datetime.datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    
    return doc