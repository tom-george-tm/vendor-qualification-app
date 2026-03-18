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

class ProjectSummary(BaseModel):
    applicant_name: str
    policy_number: Optional[str] = None
    applicant_age: Optional[int] = None
    patient_gender: Optional[str] = None
    medical_case: str
    diagnosis: Optional[str] = None
    procedure: Optional[str] = None
    hospital_name: str
    hospital_location: Optional[str] = None
    claimed_amount: Optional[float] = None
    analysis_timestamp: str
    documents_analyzed: int

class OverallAssessment(BaseModel):
    readiness_score: int
    risk_level: str
    submission_status: str
    documents_analyzed: int
    critical_issues: int
    moderate_issues: int
    minor_issues: int

class ChecklistResultItem(BaseModel):
    criterion: str
    risk_level: str
    explanation: Optional[str] = None

class DocumentAnalysisDetail(BaseModel):
    document_type: str
    document_status: str
    risk_level: str
    decision_reasoning: str
    policy_number: Optional[str] = None
    applicant_name: Optional[str] = None
    applicant_age: Optional[int] = None
    patient_gender: Optional[str] = None
    hospital_name: Optional[str] = None
    hospital_location: Optional[str] = None
    diagnosis: Optional[str] = None
    procedure: Optional[str] = None
    claimed_amount: Optional[float] = None
    recommendations: List[str]
    checklist_results: List[ChecklistResultItem]
    filename: Optional[str] = None
    db_id: Optional[str] = None

class CrossDocumentValidation(BaseModel):
    validation_type: str
    documents_compared: List[str]
    result: str
    details: str
    risk_level: str

class ActionItem(BaseModel):
    priority: int
    action: str
    description: str

class ComprehensiveReadinessResponse(BaseModel):
    project_summary: ProjectSummary
    overall_assessment: OverallAssessment
    ai_summary: Dict[str, str]
    document_analysis: List[DocumentAnalysisDetail]
    cross_document_validation: List[CrossDocumentValidation]
    prioritized_action_items: List[ActionItem]
    next_steps: List[str]
    id: Optional[str] = Field(None, alias="_id")

    class Config:
        populate_by_name = True
