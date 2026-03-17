import json
from datetime import date


# ---------------------------------------------------------------------------
# Checklists by document type
# ---------------------------------------------------------------------------

CHECKLISTS = {
    "Bill": {
        "dated_within_3_months": "Is the bill dated within the last 3 months? (Expected: Within 3 months)",
        "full_name_present": "Does the bill include the full name of the insured? (Expected: Full name present)",
        "amount_due_specified": "Is the amount due clearly specified? (Expected: Amount specified)",
        "billing_address_present": "Is the billing address present? (Expected: Address present)",
    },
    "Blood report": {
        "dated_within_1_month": "Is the blood report dated within the last 1 month? (Expected: Within 1 month)",
        "full_name_present": "Does the report include the full name of the insured? (Expected: Full name present)",
        "test_results_present": "Are the test results clearly presented? (Expected: Results present)",
        "lab_name_present": "Is the laboratory name and accreditation present? (Expected: Lab name present)",
    },
    "Medical report": {
        "dated_within_1_month": "Is the medical report dated within the last 1 month? (Expected: Within 1 month)",
        "full_name_present": "Does the report include the full name of the insured? (Expected: Full name present)",
        "diagnosis_present": "Is there a clear diagnosis or medical condition stated? (Expected: Diagnosis present)",
        "physician_signature": "Is the report signed by a licensed physician? (Expected: Physician signature present)",
    }
}

def build_analysis_prompt(doc_type: str) -> str:
    """
    Build the system prompt that instructs the AI workflow
    to evaluate a document and extract the layout requested by the user.
    """
    today_str = date.today().strftime("%d-%b-%Y")  # e.g. "17-Mar-2026"

    criteria = CHECKLISTS.get(doc_type, {})
    items = "\n".join(f"- {label}" for label in criteria.values())
    checklist_text = f"{doc_type} Checklist:\n{items}"

    prompt = f"""You are an expert in insurance case analysis and document validation.

TODAY'S DATE: {today_str}
Use this as the reference date for ALL date-window checks (e.g. "within 1 month" means
the document date must fall between {today_str} minus 30 days and {today_str}).
Do NOT treat any date that is on or before {today_str} as a future date.

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
  "applicant_name": "<Extract from documents>",
  "medical_case": "<Extract from documents>",
  "applicant_age": <number or null>,
  "hospital_name": "<Extract from documents>",
  "analysis_timestamp": "<current_iso_timestamp>",
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
    today_str = date.today().strftime("%d-%b-%Y")

    prompt = f"""You are a Senior Insurance Claim Risk Manager and AI insurance Claim Readiness Assistant for insurance claim applications.
You will be provided with a JSON containing multiple document analysis results for an insurance claim.
Your task is to aggregate these results and provide a comprehensive claim readiness assessment.

TODAY'S DATE: {today_str}
Use this as the reference date when reasoning about whether document dates are valid, recent, or expired.
Do NOT treat any date on or before {today_str} as a future date.

The mandatory documents for this claim type are:
- Bill
- Blood report
- Medical report

CRITICAL REQUIREMENTS:
1. If any of the three mandatory documents are missing from the input, set the Overall Risk Level to "High" or "Critical".
2. Perform "Cross-Document Validation": Check for consistency between documents (e.g., applicant name in Bill vs Medical report, dates, hospital names).
3. Identify "Critical", "Moderate", and "Minor" issues across all documentation.
4. Provide a prioritized action list and immediate next steps.
5. IMPORTANT — Medical case cross-validation: A document may state the DIAGNOSIS (e.g. "Acute Appendicitis") while another states the PROCEDURE/TREATMENT (e.g. "Laparoscopic Appendectomy"). These are complementary terms for the same clinical event and must NOT be flagged as a mismatch. Only flag a mismatch if the underlying medical condition is genuinely different across documents.
6. IMPORTANT — Date validation: use TODAY'S DATE as the only reference. A bill dated after the admission date and on or before today is valid. Do NOT flag a date as future if it is on or before {today_str}.

Return ONLY the following JSON structure, exact keys and data types, no markdown blocks:

{{
  "claim_summary": {{
    "applicant_name": "<Extract from documents>",
    "medical_case": "<Extract from documents>",
    "applicant_age": <number or null>,
    "hospital_name": "<Extract from documents>",
    "analysis_timestamp": "<current_iso_timestamp>",
    "documents_analyzed": <count>
  }},
  "overall_assessment": {{
    "readiness_score": <integer 0-100 indicating likelihood of claim approval>,
    "risk_level": "Low" | "Moderate" | "High" | "Critical",
    "submission_status": "Ready" | "Not Ready",
    "documents_analyzed": <count>,
    "critical_issues": <count>,
    "moderate_issues": <count>,
    "minor_issues": <count>
  }},
  "ai_summary": {{
    "summary_text": "<executive summary of insurance claim readiness, highlighting key risks and inconsistencies>"
  }},
  "document_analysis": [
     // Include the individual document results here, slightly refined if needed
  ],
  "cross_document_validation": [
    {{
      "validation_type": "<e.g., 'Applicant Name Consistency', 'Date Consistency', 'Hospital Name Consistency'>",
      "documents_compared": ["Doc A", "Doc B"],
      "result": "Match" | "Mismatch" | "Warning",
      "details": "<explanation of the cross-check result>",
      "risk_level": "Low" | "Moderate" | "High" | "Critical"
    }}
  ],
  "prioritized_action_items": [
    {{
      "priority": <integer 1, 2, 3...>,
      "action": "<short title of action>",
      "description": "<detailed instruction on how to resolve the issue>"
    }}
  ],
  "next_steps": [
    "<string step 1>",
    "<string step 2>"
  ]
}}"""
    return prompt
