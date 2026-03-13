import json

"""
UAE Regulatory Permit Document Analysis
========================================
Checklists and prompt builder for AI-powered document evaluation
formatting specifically into the required ChatResponse structure.
"""

# ---------------------------------------------------------------------------
# Checklists by document type
# ---------------------------------------------------------------------------

CHECKLISTS = {
    "EIA": {
        "baseline_environmental_study": "Does it contain a baseline environmental study? (Expected: Yes)",
        "marine_coastal_impact": "Is there a marine/coastal impact section? (Expected: Yes)",
        "impact_radius_ecosystems": "Is the impact radius for sensitive ecosystems specified? (Expected: Specified)",
        "environmental_management_plan": "Is an Environmental Management Plan included? (Expected: Yes)",
        "study_dated_18_months": "Is the study dated within the last 18 months? (Expected: Within 18 months)",
    },
    "Grid Assessment": {
        "load_data_2023_or_later": "Does it reference load data from 2023 or later? (Expected: Post-2023)",
        "named_substation": "Is a named substation connection point referenced? (Expected: Named)",
        "fault_level_analysis": "Is a fault level analysis present? (Expected: Present)",
        "grid_operator_named": "Is the grid operator authority named? (Expected: DEWA/SEWA/ADWEA)",
    },
    "EPC Contract": {
        "arabic_translation": "Is an Arabic translation attached or referenced? (Expected: Yes)",
        "performance_bonds": "Are performance bonds specified? (Expected: Specified)",
        "signed_both_parties": "Is the contract signed by both parties? (Expected: Yes)",
    },
    "GCAA NOC": {
        "document_present": "Is the document present? (if absent, critical blocker) (Expected: Present)",
        "approval_within_12_months": "Is the approval date within 12 months? (Expected: Within 12 months)",
        "correct_gps_coordinates": "Does it cover the correct GPS coordinates? (Expected: Correct coordinates)",
    },
    "Bank Letter": {
        "dated_within_6_months": "Is the document dated within 6 months? (Expected: Within 6 months)",
        "confirms_full_financing": "Does it confirm the full project financing amount? (Expected: Full amount confirmed)",
        "official_letterhead": "Is it on official bank letterhead? (Expected: Official letterhead)",
    },
}

def build_analysis_prompt(doc_type: str) -> str:
    """
    Build the system prompt that instructs the AI workflow
    to evaluate a document and extract the layout requested by the user.
    """

    criteria = CHECKLISTS.get(doc_type, {})
    items = "\n".join(f"- {label}" for label in criteria.values())
    checklist_text = f"{doc_type} Checklist:\n{items}"

    prompt = f"""You are an expert Regulatory & Technical Auditor. 

You will receive an image of a document ({doc_type}).
Your job is to:
1. Evaluate the document against its specific checklist below.
2. Determine any deviations or adherence for each criterion.
3. Assign a document-level risk level ("Low", "Moderate", "High", "Critical").
4. Provide a detailed decision reasoning.
5. Provide actionable recommendations.

---

{checklist_text}

---

Return ONLY the following JSON structure, exact keys and data types, no markdown blocks:

{{
  "document_type": "{doc_type}",
  "document_status": "Reviewed",
  "risk_level": "Low" | "Moderate" | "High" | "Critical",
  "decision_reasoning": "<paragraph explaining the evaluation result>",
  "recommendations": [
    "<string actionable recommendation 1>",
    "<string actionable recommendation 2>"
  ],
  "checklist_results": [
    {{
      "criterion": "<the checklist item question or description>",
      "risk_level": "NA" | "minor" | "moderate" | "critical",
      "explanation": "<short explanation for the risk level if not NA>"
    }}
  ]
}}"""
    return prompt

def build_aggregator_prompt() -> str:
    """
    Build the system prompt for the aggregator agent.
    """
    prompt = """You are a Senior Project Risk Manager.
You will be provided with a JSON containing multiple document analysis results for a project.
Your task is to aggregate these results and provide a comprehensive project readiness assessment.

The mandatory documents for this project type are:
- EIA (Environmental Impact Assessment)
- Grid Assessment
- EPC Contract
- GCAA NOC
- Bank Letter

CRITICAL REQUIREMENTS:
1. If any of the five mandatory documents are missing from the input, set the Overall Risk Level to "High" or "Critical".
2. Perform "Cross-Document Validation": Check for consistency between documents (e.g., project capacity in EIA vs EPC Contract, locations, dates).
3. Identify "Critical", "Moderate", and "Minor" issues across all documentation.
4. Provide a prioritized action list and immediate next steps.

Return ONLY the following JSON structure, exact keys and data types, no markdown blocks:

{
  "project_summary": {
    "project_name": "<Extract from documents or use 'Project Al Noor' if unknown>",
    "project_type": "<Extract from documents>",
    "capacity_mw": <number or null>,
    "location": "<Extract from documents>",
    "analysis_timestamp": "<current_iso_timestamp>",
    "documents_analyzed": <count>
  },
  "overall_assessment": {
    "readiness_score": <integer 0-100 indicating likelihood of permit approval>,
    "risk_level": "Low" | "Moderate" | "High" | "Critical",
    "submission_status": "Ready" | "Not Ready",
    "documents_analyzed": <count>,
    "critical_issues": <count>,
    "moderate_issues": <count>,
    "minor_issues": <count>
  },
  "ai_summary": {
    "summary_text": "<executive summary of project readiness, highlighting key risks and inconsistencies>"
  },
  "document_analysis": [
     // Include the individual document results here, slightly refined if needed
  ],
  "cross_document_validation": [
    {
      "validation_type": "<e.g., Capacity Consistency, Location Match, Date Validity>",
      "documents_compared": ["Doc A", "Doc B"],
      "result": "Match" | "Mismatch" | "Warning",
      "details": "<explanation of the cross-check result>",
      "risk_level": "Low" | "Moderate" | "High" | "Critical"
    }
  ],
  "prioritized_action_items": [
    {
      "priority": <integer 1, 2, 3...>,
      "action": "<short title of action>",
      "description": "<detailed instruction on how to resolve the issue>"
    }
  ],
  "next_steps": [
    "<string step 1>",
    "<string step 2>"
  ]
}"""
    return prompt
