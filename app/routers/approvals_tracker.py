from fastapi import APIRouter
import json
import logging
from openai import AsyncAzureOpenAI

from app.constant.approvals_tracker import approvals_tracker_data
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

openai_client = AsyncAzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
)

@router.get("/cards")
async def get_cards():
    return approvals_tracker_data

@router.get("/ai-summary")
async def get_ai_summary():
    system_prompt = (
        "You are an expert project risk analyst.\n"
        "Analyze the provided approvals tracker data and return a response structured in these exact 4 sections in markdown:\n\n"
        "## Operational Summary\n"
        "(paragraph summary)\n\n"
        "## Risk Highlights\n"
        "- (bullet points)\n\n"
        "## Recommendations\n"
        "- (bullet points)\n\n"
        "## Forecast\n"
        "(paragraph forecast)"
    )
    
    user_content = json.dumps(approvals_tracker_data)

    try:
        response = await openai_client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_MINI,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        ai_summary = response.choices[0].message.content
        return {
            "input_data": approvals_tracker_data,
            "ai_summary": ai_summary
        }
    except Exception as e:
        logger.error("OpenAI API call failed: %s", e)
        return {
            "input_data": approvals_tracker_data,
            "ai_summary": f"Failed to generate summary: {str(e)}"
        }
