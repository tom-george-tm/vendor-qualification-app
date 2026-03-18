from fastapi import APIRouter
import json
import os

_here = os.path.dirname(__file__)
with open(os.path.join(_here, "../constant/dashboard.json"), "r") as f:
    dashboard_data = json.load(f)

router = APIRouter()

@router.get("/")
async def get_dashboard_data():
    """
    Returns the comprehensive dashboard portfolio data.
    """
    return dashboard_data
