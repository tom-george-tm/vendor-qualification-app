from fastapi import APIRouter
from app.constant.dashboard import dashboard_data

router = APIRouter()

@router.get("/")
async def get_dashboard_data():
    """
    Returns the comprehensive dashboard portfolio data.
    """
    return dashboard_data
