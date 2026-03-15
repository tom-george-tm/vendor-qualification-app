from fastapi import APIRouter
import json
import logging
from openai import AsyncAzureOpenAI

from app.constant.approvals_tracker import approvals_tracker_data
from app.constant.dashboard import dashboard_data
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
        "Analyze the provided dashboard data, which includes a portfolio of energy and infrastructure projects in the UAE.\n"
        "Return a comprehensive summary of the entire dashboard structured in these exact 4 sections in markdown:\n\n"
        "## Operational Summary\n"
        "(paragraph summary of the portfolio health, total value, and capacity)\n\n"
        "## Risk Highlights\n"
        "- (bullet points identifying specific projects that are Blocked or At Risk, and overdue NOCs)\n\n"
        "## Recommendations\n"
        "- (bullet points for actionable steps to mitigate risks and move projects forward)\n\n"
        "## Forecast\n"
        "(paragraph forecast of expected progress and upcoming critical deadlines)"
    )
    
    user_content = json.dumps(dashboard_data)

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
            "ai_summary": ai_summary
        }
    except Exception as e:
        logger.error("OpenAI API call failed: %s", e)
        return {
            "ai_summary": f"Failed to generate summary: {str(e)}"
        }
