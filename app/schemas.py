"""
Pydantic models for validating structured document analysis responses
and aggregated project-level results.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

class Deviation(BaseModel):
    item: str
    expected: str
    actual: str
    classification: Literal["minor", "moderate", "critical"]

class WorkflowStatus(BaseModel):
    tool_executed: str = "approve_submission"
    execution_status: str = "SUCCESS"

class ChatResponseFormat(BaseModel):
    """The specific chat response format requested by the user."""
    validation_score: int
    vendor_rating: float
    composite_risk_score: int
    decision: Literal["APPROVE", "REJECT"]
    deviations: List[Deviation]
    decision_reasoning: str
    recommendations: List[str]
    workflow: WorkflowStatus = Field(default_factory=WorkflowStatus)

# Legacy Support models
class NextStep(BaseModel):
    fix: str
    where: str
    timeline: str
    tip: str

class ChecklistResult(BaseModel):
    criteria_id: str
    criteria_label: str
    status: Literal["PASS", "FAIL", "PARTIAL"]
    risk_level: Literal["none", "minor", "moderate", "critical"]
    finding: str
    recommendation: Optional[NextStep] = None

class DocumentAnalysisResponse(BaseModel):
    document_type: str
    checklist_results: List[ChecklistResult]
    document_summary: str
    critical_issues: List[str] = Field(default_factory=list)
    document_score: int = Field(ge=0, le=100)

class ProjectReadinessResponse(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    project_risk_level: Literal["low", "medium", "high", "critical"]
    summary: str
    prioritized_action_list: List[NextStep]
    next_steps_summary: str
    document_analyses: List[DocumentAnalysisResponse]
