from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.get("/dashboard")
async def get_dashboard() -> Dict[str, Any]:
    """
    Mock data for 'The Approvals Tracker Dashboard' live table.
    """
    return {
        "active_applications": [
            {
                "project_name": "RAK Offshore Solar",
                "type": "Renewable Energy",
                "value": "AED 8.2B",
                "stage": "2 of 6",
                "status": "Blocked",
                "days_since_update": 5,
                "assigned_officer": "Ahmed Al Maktoum"
            },
            {
                "project_name": "Dubai Marina Expansion Phase 3",
                "type": "Commercial Real Estate",
                "value": "AED 1.4B",
                "stage": "4 of 6",
                "status": "On Track",
                "days_since_update": 1,
                "assigned_officer": "Sara Al Hashimi"
            },
            {
                "project_name": "Abu Dhabi Logistics Hub",
                "type": "Infrastructure",
                "value": "AED 450M",
                "stage": "1 of 6",
                "status": "At Risk",
                "days_since_update": 12,
                "assigned_officer": "Omar Tariq"
            }
        ]
    }

@router.get("/alerts")
async def get_alerts() -> Dict[str, Any]:
    """
    Mock data for the 'AI Alert Panel'.
    """
    return {
        "alerts": [
            {
                "type": "Document gap flagged",
                "severity": "Critical",
                "project": "RAK Offshore Solar",
                "title": "EIA Rejected",
                "description": "Environmental Impact Assessment rejected by MOCCAE on 8 March. Marine ecosystem impact study insufficient. Without resubmission within 30 days, project loses site reservation — estimated AED 8.2B exposure. AI has identified 3 similar approved EIAs that can serve as templates.",
                "action_button": "View Project"
            },
            {
                "type": "SLA breach detected",
                "severity": "Moderate",
                "project": "Abu Dhabi Logistics Hub",
                "title": "NOC Delayed > 10 days",
                "description": "DEWA grid connection assessment is currently 12 days past the standard 14-day SLA. High probability of delaying the Q3 general contractor start date.",
                "action_button": "Draft Letter"
            },
            {
                "type": "Regulatory update",
                "severity": "Minor",
                "project": "Global",
                "title": "New GCAA Drone Regulations",
                "description": "GCAA updated drone survey requirements yesterday. active projects using drone topography scans in staging phase require form GCAA-118.",
                "action_button": "Review Impact"
            }
        ]
    }

@router.get("/briefing")
async def get_briefing() -> Dict[str, Any]:
    """
    Mock data for the 'Briefing Generator'.
    """
    return {
        "briefing_note": {
            "date": "13 March 2026",
            "executive_summary": "Pipeline is largely healthy (14 projects on track). 2 projects require immediate executive intervention. Estimated at-risk capital is AED 8.65B.",
            "key_blockers": [
                "MOCCAE marine assessment requirements tightening (affecting RAK Solar).",
                "DEWA processing times increased by 30% month-over-month."
            ],
            "recommended_actions": [
                "Schedule escalation call with MOCCAE Director regarding Marine Guidelines annex C.",
                "Reallocate 2 engineers to finalise the DEWA load data update for Abu Dhabi Logistics Hub."
            ]
        }
    }
