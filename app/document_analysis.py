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

def build_analysis_prompt() -> str:
    """
    Build the system prompt that instructs the AI workflow
    to evaluate a document and exact the layout requested by the user.
    """

    checklist_sections = []
    for doc_type, criteria in CHECKLISTS.items():
        items = "\n".join(f"- {label}" for label in criteria.values())
        checklist_sections.append(f"{doc_type} Checklist:\n{items}")
    checklists_text = "\n\n".join(checklist_sections)

    prompt = f"""You are an expert Regulatory & Technical Auditor. 

You will receive an image of a document (EIA, Grid Assessment, EPC Contract, GCAA NOC, or Bank Letter).
Your job is to:
1. Identify the document type based on its content.
2. Evaluate the document against its specific checklist below.
3. Determine any deviations (where actual content fails to meet the expected criteria).
4. Assign a composite risk score (0-100, where 0 is no risk, higher is more risky based on deviations).
5. Assign a vendor rating out of 5.0 indicating document quality.
6. Provide a decision ('APPROVE' or 'REJECT') based on critical deviations.

---

CHECKLISTS BY TYPE:

{checklists_text}

---

Return ONLY the following JSON structure, exact keys and data types, no markdown blocks:

{{
  "validation_score": <integer 0-100>,
  "vendor_rating": <float 0.0-5.0>,
  "composite_risk_score": <integer 0-100>,
  "decision": "APPROVE" | "REJECT",
  "deviations": [
    {{
      "item": "<checklist item name>",
      "expected": "<what was required>",
      "actual": "<what was found in the document>",
      "classification": "minor" | "moderate" | "critical"
    }}
  ],
  "decision_reasoning": "<paragraph explaining the decision, summarizing the rating, validation score, and deviations>",
  "recommendations": [
    "<string actionable recommendation 1>",
    "<string actionable recommendation 2>"
  ],
  "workflow": {{
    "tool_executed": "approve_submission",
    "execution_status": "SUCCESS"
  }}
}}"""

    return prompt
