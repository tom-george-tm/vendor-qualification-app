import json
from datetime import date, timedelta


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
    today = date.today()
    today_str        = today.strftime("%d-%b-%Y")            # e.g. "19-Mar-2026"
    one_month_ago    = (today - timedelta(days=30)).strftime("%d-%b-%Y")
    three_months_ago = (today - timedelta(days=90)).strftime("%d-%b-%Y")

    criteria = CHECKLISTS.get(doc_type, {})
    items = "\n".join(f"- {label}" for label in criteria.values())
    checklist_text = f"{doc_type} Checklist:\n{items}"

    prompt = f"""You are an expert in insurance case analysis and document validation.

TODAY'S DATE: {today_str}

STRICT DATE RULES — follow these exactly, no exceptions:
1. A date is VALID (not in the future) if it is on or before {today_str}.
   Do NOT flag any date that is on or before {today_str} as a future date.
2. "Within 3 months" means the document date is between {three_months_ago} and {today_str} (inclusive).
   Any date from {three_months_ago} up to and including {today_str} PASSES this check.
3. "Within 1 month" means the document date is between {one_month_ago} and {today_str} (inclusive).
   Any date from {one_month_ago} up to and including {today_str} PASSES this check.
4. Dates inside a document that describe events BEFORE today (e.g. surgery performed on a past date,
   admission date, procedure date) are NOT future dates — evaluate them against the rules above.
5. When in doubt, compare day-month-year numerically. {today_str} is the ceiling; anything equal
   to or earlier than this date is a past or present date, NEVER a future date.

CHECKLIST EVALUATION RULES:
- Only mark a criterion as "Fail" if you have EXPLICIT, CLEAR, VISIBLE evidence in the document that the requirement is not met.
- Only mark a criterion as "Warning" if the information is partially present or ambiguous.
- When in doubt, mark as "Pass" — do NOT invent or assume failures.
- Do NOT fail a criterion because you "cannot see" something that is simply off-screen or in a table format.
- A physician signature shown as "[SIGNED]", "[SIGNED & STAMPED]", or any textual representation counts as a valid signature.
- A lab accreditation number (e.g. NABL-XXXX) counts as lab accreditation present.

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

  "policy_number": "<Extract from documents or null>",
  "applicant_name": "<Extract from documents>",
  "applicant_age": <number or null>,
  "patient_gender": "Male" | "Female" | "Other" | null,
  "medical_case": "<Extract from documents>",
  "diagnosis": "<Extract detailed diagnosis if available>",
  "procedure": "<Extract detailed medical procedure if available>",
  "hospital_name": "<Extract from documents>",
  "hospital_location": "<Extract city/location or null>",
  "claimed_amount": <number or null if Bill>,
  "analysis_timestamp": "<current_iso_timestamp>",
  "document_type": "{doc_type}",
  "document_status": "Reviewed",
  "risk_level": "Low" | "Moderate" | "High" | "Critical",
  "decision_reasoning": "<paragraph explaining the evaluation result, EXPLICITLY listing all issues or discrepancies found>",
  "recommendations": [
    "<string actionable recommendation 1>",
    "<string actionable recommendation 2>"
  ],
  "checklist_results": [
    {{
      "criterion": "<the checklist item question or description>",
      "status": "Pass" | "Fail" | "Warning",
      "risk_level": "NA" | "minor" | "moderate" | "critical",
      "explanation": "<EXPLICIT detail on why it failed or passed>"
    }}
  ]
}}"""
    return prompt

def build_aggregator_prompt() -> str:
    """
    Build the system prompt for the aggregator agent.
    """
    today = date.today()
    today_str        = today.strftime("%d-%b-%Y")
    one_month_ago    = (today - timedelta(days=30)).strftime("%d-%b-%Y")
    three_months_ago = (today - timedelta(days=90)).strftime("%d-%b-%Y")

    prompt = f"""You are a Senior Insurance Claim Risk Manager and AI insurance Claim Readiness Assistant for insurance claim applications.
You will be provided with a JSON containing multiple document analysis results for an insurance claim.
Your task is to aggregate these results and provide a comprehensive claim readiness assessment.

TODAY'S DATE: {today_str}

STRICT DATE RULES — apply these when re-evaluating any date issues flagged by individual document analyses:
1. A date is valid (not future) if it is on or before {today_str}. NEVER call a date "future" if it is ≤ {today_str}.
2. "Within 3 months" = document date is between {three_months_ago} and {today_str} (inclusive). Dates in this range PASS.
3. "Within 1 month"  = document date is between {one_month_ago} and {today_str} (inclusive). Dates in this range PASS.
4. If an individual document analysis INCORRECTLY flagged a date that is actually within the valid range above,
   DO NOT carry that error forward. Override it: mark it as PASS in your aggregation.
5. Procedure dates, surgery dates, and admission dates that fall before {today_str} are historical facts,
   not future events — they must NEVER be flagged as invalid future dates.

ANTI-HALLUCINATION RULES — critical:
- Only list an issue in `all_detected_issues` if it is EXPLICITLY present in the individual document analyses provided.
- Do NOT invent new issues that were not flagged by the per-document analysis.
- Do NOT flag a document as "missing" if it appears in the input JSON with a valid `document_type`.
- If all 3 mandatory documents (Bill, Blood report, Medical report) are present in the input, do NOT say any are missing.
- Consistency rule: if `readiness_score` >= 80 and no critical issues and all 3 docs are present, `final_suggestion` MUST be "APPROVE" and `submission_status` MUST be "Ready".

The mandatory documents for this claim type are:
- Bill
- Blood report
- Medical report

CRITICAL REQUIREMENTS:
1. If any of the three mandatory documents are missing from the input, set the Overall Risk Level to "High" or "Critical".
2. Perform "Cross-Document Validation": Check for consistency between documents (e.g., applicant name in Bill vs Medical report, dates, hospital names).
3. Identify ALL "Critical", "Moderate", and "Minor" issues across all documentation.
4. Determine Coverage Status: Based on document validity and completeness, specify if the claim appears "Fully Covered", "Partially Covered" (estimate percentage if possible), or "Not Covered".
5. Provide a prioritized action list and immediate next steps.
6. IMPORTANT — Medical case cross-validation: A document may state the DIAGNOSIS (e.g. "Acute Appendicitis") while another states the PROCEDURE/TREATMENT (e.g. "Laparoscopic Appendectomy"). These are complementary terms for the same clinical event and must NOT be flagged as a mismatch. Only flag a mismatch if the underlying medical condition is genuinely different across documents.
7. IMPORTANT — Date validation: use the STRICT DATE RULES above. A bill dated on or before {today_str} is valid.
   Any date between {three_months_ago} and {today_str} passes the 3-month check.
   Any date between {one_month_ago} and {today_str} passes the 1-month check.
   If a sub-document analysis flagged a valid date as an error, CORRECT it in your aggregation — do not inherit the mistake.

--- DECISION RULES (follow these EXACTLY) ---

Use these deterministic rules for `final_suggestion`:
- Set "APPROVE" ONLY when ALL of these are true:
  (a) All 3 mandatory documents are present
  (b) No policy exclusions or coverage violations are found
  (c) All checklist items pass or have only minor issues
  (d) readiness_score is >= 80
- Set "REJECT" ONLY when:
  (a) A policy exclusion explicitly applies (e.g. cosmetic procedure, elective surgery), OR
  (b) The treatment/procedure is explicitly not covered under the policy terms
- Set "MORE_INFO_NEEDED" in ALL other cases, including:
  (a) Any mandatory document is missing
  (b) Critical or moderate checklist failures exist that could be corrected
  (c) Document quality issues (e.g. missing signatures, expired dates)

Use these deterministic rules for `risk_level`:
- "Low": 0 critical issues AND 0 moderate issues
- "Moderate": 0 critical issues AND 1+ moderate issues
- "High": 1-2 critical issues OR any mandatory document missing
- "Critical": 3+ critical issues OR treatment not covered by policy

Use these deterministic rules for `submission_status`:
- "Ready": final_suggestion is "APPROVE"
- "Not Ready": final_suggestion is "REJECT" or "MORE_INFO_NEEDED"

--- END DECISION RULES ---

Return ONLY the following JSON structure, exact keys and data types, no markdown blocks:

{{
  "claim_summary": {{
    "policy_number": "<Extract from documents>",
    "applicant_name": "<Extract from documents>",
    "applicant_age": <number or null>,
    "patient_gender": "Male" | "Female" | "Other" | null,
    "medical_case": "<Extract from documents>",
    "diagnosis": "<Extract detailed diagnosis>",
    "procedure": "<Extract detailed medical procedure>",
    "hospital_name": "<Extract from documents>",
    "hospital_location": "<Extract city/location>",
    "claimed_amount": <total_sum_from_bills or null>,
    "analysis_timestamp": "<current_iso_timestamp>",
    "documents_analyzed": <count>
  }},
  "overall_assessment": {{
    "readiness_score": <integer 0-100>,
    "risk_level": "Low" | "Moderate" | "High" | "Critical",
    "submission_status": "Ready" | "Not Ready",
    "coverage_status": "Fully Covered" | "Partially Covered (X%)" | "Not Covered",
    "final_suggestion": "APPROVE" | "REJECT" | "MORE_INFO_NEEDED",
    "all_detected_issues": [
      "<EXPLICIT list of every issue found across all docs>"
    ],
    "documents_analyzed": <count>,
    "critical_issues": <count>,
    "moderate_issues": <count>,
    "minor_issues": <count>
  }},
  "ai_summary": {{
    "summary_text": "<executive summary emphasizing risks, issues, and coverage level>"
  }},
  "document_analysis": [
     // Include the individual document results here
  ],
  "cross_document_validation": [
    {{
      "validation_type": "<e.g., 'Applicant Name Consistency'>",
      "documents_compared": ["Doc A", "Doc B"],
      "result": "Match" | "Mismatch" | "Warning",
      "details": "<explanation>",
      "risk_level": "Low" | "Moderate" | "High" | "Critical"
    }}
  ],
  "prioritized_action_items": [
    {{
      "priority": <number>,
      "action": "<title>",
      "description": "<detailed instruction>"
    }}
  ],
  "next_steps": [
    "<string step 1>",
    "<string step 2>"
  ]
}}"""
    return prompt
