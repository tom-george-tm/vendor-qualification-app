"""
generate_test_claim_pdfs.py
===========================
Generates insurance claim PDFs for three test scenarios.

SCENARIOS
---------
  success    — All 3 valid docs. All checklists pass → Ready / Low risk.
  failure    — All 3 docs present but with critical defects (future date on bill,
               no physician signature on medical report, missing lab accreditation
               on blood report, patient name mismatch across docs).
  incomplete — Only 1 doc uploaded (blood report). Bill + Medical report missing.

Usage
-----
    uv run python scripts/generate_test_claim_pdfs.py <CLAIM_ID> [scenario]

    scenario defaults to "success"

    # Examples
    uv run python scripts/generate_test_claim_pdfs.py CLAIM_ID_192113 success
    uv run python scripts/generate_test_claim_pdfs.py CLAIM_ID_999001 failure
    uv run python scripts/generate_test_claim_pdfs.py CLAIM_ID_999002 incomplete

PDFs are saved inside   ./temp_uploads/<CLAIM_ID>/
"""

import os
import sys
from datetime import date, timedelta

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable,
    )
except ImportError:
    print("reportlab is required. Install it with:  pip install reportlab")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration — edit this block to change patient / hospital details
# ---------------------------------------------------------------------------

PATIENT_NAME = "Rahul Sharma"
PATIENT_DOB = "15-Aug-1990"
PATIENT_ADDRESS = "42, Green Park Lane, Chennai – 600 010, Tamil Nadu"
POLICY_NUMBER = "HI-2025-INS-004782"
CLAIM_ID_LABEL = "CLM-2025-03-00912"

HOSPITAL_NAME = "Apollo Hospitals, Chennai"
HOSPITAL_ADDRESS = "21, Greams Lane, Off Greams Road, Chennai – 600 006"
HOSPITAL_REG = "Reg. No: TN-MED-2003-00142"

LAB_NAME = "Apollo Diagnostics"
LAB_ACCREDITATION = "NABL Accredited | Lic. No: NABL-2019-LAB-4872"

DOCTOR_NAME = "Dr. Priya Anand"
DOCTOR_REG = "Reg. No: TN-MCI-20842  |  MBBS, MS (General Surgery)"

DIAGNOSIS = "Acute Appendicitis"
PROCEDURE = "Laparoscopic Appendectomy"

# Dates — all dynamic, consistent, and within valid checklist windows
TODAY = date.today()
ADMISSION_DATE = TODAY - timedelta(days=7)   # admitted 7 days ago
DISCHARGE_DATE = TODAY - timedelta(days=5)   # discharged 5 days ago
BILL_DATE      = TODAY - timedelta(days=5)   # bill issued on discharge day (>= admission)
REPORT_DATE    = TODAY - timedelta(days=5)   # all reports dated on discharge day

ADMISSION_DATE_STR = ADMISSION_DATE.strftime("%d-%b-%Y")
DISCHARGE_DATE_STR = DISCHARGE_DATE.strftime("%d-%b-%Y")
BILL_DATE_STR      = BILL_DATE.strftime("%d-%b-%Y")
REPORT_DATE_STR    = REPORT_DATE.strftime("%d-%b-%Y")

DIR = "./temp_uploads"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontSize=16,
    spaceAfter=4,
    textColor=colors.HexColor("#1a3c6e"),
)
HEADER_STYLE = ParagraphStyle(
    "SectionHeader",
    parent=styles["Heading2"],
    fontSize=11,
    spaceBefore=8,
    spaceAfter=4,
    textColor=colors.HexColor("#1a3c6e"),
)
BODY_STYLE = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=9,
    leading=14,
)
BOLD_BODY = ParagraphStyle(
    "BoldBody",
    parent=BODY_STYLE,
    fontName="Helvetica-Bold",
)
SMALL = ParagraphStyle("Small", parent=BODY_STYLE, fontSize=8, textColor=colors.grey)


def _table_style(header_bg=colors.HexColor("#1a3c6e")):
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6fb")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]
    )


def _sep():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6)


def _header_block(doc_title: str, sub: str):
    return [
        Paragraph(HOSPITAL_NAME, TITLE_STYLE),
        Paragraph(HOSPITAL_ADDRESS, SMALL),
        Paragraph(HOSPITAL_REG, SMALL),
        Spacer(1, 4 * mm),
        _sep(),
        Paragraph(doc_title, HEADER_STYLE),
        Paragraph(sub, SMALL),
        _sep(),
    ]


def _patient_info_table():
    data = [
        ["Patient Name", PATIENT_NAME, "Policy No.", POLICY_NUMBER],
        ["Date of Birth", PATIENT_DOB, "Claim ID", CLAIM_ID_LABEL],
        ["Address", PATIENT_ADDRESS, "", ""],
    ]
    t = Table(data, colWidths=[38 * mm, 62 * mm, 28 * mm, 52 * mm])
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("SPAN", (1, 2), (3, 2)),
            ]
        )
    )
    return t


# ---------------------------------------------------------------------------
# 1.  HOSPITAL BILL
# ---------------------------------------------------------------------------

def generate_bill(output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                             topMargin=15 * mm, bottomMargin=15 * mm)
    story = _header_block(
        "HOSPITAL BILL / TAX INVOICE",
        f"Bill No: APL-2026-{CLAIM_ID_LABEL}  |  Bill Date: {BILL_DATE_STR}  |  "
        f"Admission: {ADMISSION_DATE_STR}  |  Discharge: {DISCHARGE_DATE_STR}"
    )

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Patient Information", HEADER_STYLE))
    story.append(_patient_info_table())
    story.append(Spacer(1, 4 * mm))

    # Diagnosis summary on bill
    story.append(Paragraph("Clinical Summary", HEADER_STYLE))
    clin = [
        ["Diagnosis:", f"{DIAGNOSIS} (ICD-10: K35.80)"],
        ["Procedure:", PROCEDURE],
        ["Treating Surgeon:", f"{DOCTOR_NAME}  |  {DOCTOR_REG}"],
    ]
    ct = Table(clin, colWidths=[40 * mm, 140 * mm])
    ct.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(ct)
    story.append(Spacer(1, 3 * mm))

    # Charge breakdown
    story.append(Paragraph("Charge Breakdown", HEADER_STYLE))
    charges = [
        ["S.No", "Description", "Qty", "Unit Rate (₹)", "Amount (₹)"],
        ["0", f"Diagnosis: {DIAGNOSIS} | Procedure: {PROCEDURE}", "", "", ""],  # context row
        ["1", "Room & Nursing Charges (2 nights × ₹5,000/night)", "2", "5,000.00", "10,000.00"],
        ["2", "Laparoscopic Appendectomy – Surgical Fee", "1", "60,000.00", "60,000.00"],
        ["3", "Anaesthesia Charges", "1", "8,000.00", "8,000.00"],
        ["4", "OT Consumables & Disposables", "1", "7,500.00", "7,500.00"],
        ["5", "Diagnostic Tests (Pre-op CBC, ECG, X-Ray)", "1", "4,500.00", "4,500.00"],
        ["6", "Prescribed Medicines & IV Fluids", "1", "6,000.00", "6,000.00"],
        ["7", "Ambulance Services", "1", "2,000.00", "2,000.00"],
        ["", "", "", "Sub-Total (₹)", "98,000.00"],
        ["", "", "", "CGST @ 5% (₹)", "4,900.00"],
        ["", "", "", "SGST @ 5% (₹)", "4,900.00"],
        ["", "", "", "TOTAL AMOUNT DUE (₹)", "1,07,800.00"],
    ]
    t = Table(charges, colWidths=[12 * mm, 82 * mm, 12 * mm, 30 * mm, 30 * mm])
    ts = _table_style()
    ts.add("FONTNAME", (0, len(charges) - 4), (-1, len(charges) - 1), "Helvetica-Bold")
    ts.add("BACKGROUND", (0, len(charges) - 1), (-1, len(charges) - 1), colors.HexColor("#d4edda"))
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    # Payment summary
    story.append(Paragraph("Payment & Billing Address", HEADER_STYLE))
    pay_data = [
        ["Billing Address:", PATIENT_ADDRESS],
        ["Payment Mode:", "Cashless (Insurance)"],
        ["Amount Due from Insurer:", "₹1,07,800.00"],
        ["Amount Payable by Patient:", "₹0.00  (Cashless approved)"],
    ]
    pt = Table(pay_data, colWidths=[55 * mm, 125 * mm])
    pt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(pt)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "Authorised Signatory: ___________________________   "
        f"Date: {BILL_DATE_STR}   Seal: [APOLLO HOSPITALS]",
        SMALL,
    ))

    doc.build(story)
    print(f"  ✅ Bill generated          →  {output_path}")


# ---------------------------------------------------------------------------
# 2.  BLOOD REPORT
# ---------------------------------------------------------------------------

def generate_blood_report(output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                             topMargin=15 * mm, bottomMargin=15 * mm)
    story = []

    # Lab header
    story.append(Paragraph(LAB_NAME, TITLE_STYLE))
    story.append(Paragraph("Branch: Anna Nagar, Chennai – 600 040", SMALL))
    story.append(Paragraph(LAB_ACCREDITATION, SMALL))
    story.append(Spacer(1, 3 * mm))
    story.append(_sep())
    story.append(Paragraph("HAEMATOLOGY & BIOCHEMISTRY REPORT", HEADER_STYLE))
    story.append(Paragraph(
        f"Report Date: {REPORT_DATE_STR}  |  Collection Date: {REPORT_DATE_STR}  |  "
        f"Ref ID: APD-2026-{CLAIM_ID_LABEL}",
        SMALL,
    ))
    story.append(_sep())
    story.append(Spacer(1, 2 * mm))

    story.append(_patient_info_table())
    story.append(Spacer(1, 4 * mm))

    # CBC
    story.append(Paragraph("Complete Blood Count (CBC)", HEADER_STYLE))
    cbc = [
        ["Test", "Result", "Reference Range", "Units", "Status"],
        ["Haemoglobin (Hb)", "13.8", "13.0 – 17.0", "g/dL", "Normal"],
        ["RBC Count", "4.85", "4.50 – 5.90", "×10⁶/μL", "Normal"],
        ["WBC (Total Leukocyte Count)", "11.2", "4.0 – 11.0", "×10³/μL", "Normal ↑"],
        ["Neutrophils", "72", "40 – 75", "%", "Normal"],
        ["Lymphocytes", "20", "20 – 45", "%", "Normal"],
        ["Monocytes", "6", "2 – 10", "%", "Normal"],
        ["Eosinophils", "2", "1 – 6", "%", "Normal"],
        ["Platelet Count", "2.45", "1.50 – 4.00", "×10⁵/μL", "Normal"],
        ["Haematocrit (PCV)", "41.2", "40.0 – 52.0", "%", "Normal"],
        ["MCV", "85.0", "80.0 – 96.0", "fL", "Normal"],
        ["MCH", "28.5", "27.0 – 33.0", "pg", "Normal"],
        ["MCHC", "33.5", "31.5 – 36.0", "g/dL", "Normal"],
    ]
    t = Table(cbc, colWidths=[62 * mm, 22 * mm, 38 * mm, 18 * mm, 22 * mm])
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 3 * mm))

    # LFT
    story.append(Paragraph("Liver Function Tests (LFT)", HEADER_STYLE))
    lft = [
        ["Test", "Result", "Reference Range", "Units", "Status"],
        ["Total Bilirubin", "0.8", "0.2 – 1.2", "mg/dL", "Normal"],
        ["SGOT (AST)", "28", "10 – 40", "U/L", "Normal"],
        ["SGPT (ALT)", "31", "7 – 56", "U/L", "Normal"],
        ["Alkaline Phosphatase", "74", "44 – 147", "U/L", "Normal"],
        ["Total Protein", "7.2", "6.3 – 8.2", "g/dL", "Normal"],
        ["Albumin", "4.1", "3.5 – 5.0", "g/dL", "Normal"],
    ]
    t2 = Table(lft, colWidths=[62 * mm, 22 * mm, 38 * mm, 18 * mm, 22 * mm])
    t2.setStyle(_table_style())
    story.append(t2)
    story.append(Spacer(1, 3 * mm))

    # RFT
    story.append(Paragraph("Renal Function Tests (RFT)", HEADER_STYLE))
    rft = [
        ["Test", "Result", "Reference Range", "Units", "Status"],
        ["Blood Urea Nitrogen", "14.2", "7.0 – 20.0", "mg/dL", "Normal"],
        ["Serum Creatinine", "0.92", "0.74 – 1.35", "mg/dL", "Normal"],
        ["eGFR", "98", ">60", "mL/min/1.73m²", "Normal"],
        ["Uric Acid", "5.1", "3.5 – 7.2", "mg/dL", "Normal"],
        ["Sodium (Na⁺)", "138", "136 – 145", "mEq/L", "Normal"],
        ["Potassium (K⁺)", "4.2", "3.5 – 5.0", "mEq/L", "Normal"],
    ]
    t3 = Table(rft, colWidths=[62 * mm, 22 * mm, 38 * mm, 18 * mm, 22 * mm])
    t3.setStyle(_table_style())
    story.append(t3)
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph(
        "Interpretation: All results are within normal reference ranges. "
        "Mild leukocytosis (WBC 11.2) is consistent with the post-operative inflammatory response "
        f"following {PROCEDURE} and does not indicate systemic infection.",
        BODY_STYLE,
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        f"Reported by: Dr. S. Meenakshi, MD (Pathology)  |  {LAB_NAME}  |  "
        f"Report Date: {REPORT_DATE_STR}",
        SMALL,
    ))
    story.append(Paragraph(
        f"Digital Signature: [SIGNED]  |  {LAB_ACCREDITATION}",
        SMALL,
    ))

    doc.build(story)
    print(f"  ✅ Blood report generated  →  {output_path}")


# ---------------------------------------------------------------------------
# 3.  MEDICAL / SURGICAL REPORT
# ---------------------------------------------------------------------------

def generate_medical_report(output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                             topMargin=15 * mm, bottomMargin=15 * mm)
    story = _header_block(
        "SURGICAL DISCHARGE & MEDICAL REPORT",
        f"Report Date: {REPORT_DATE_STR}  |  "
        f"Admission: {ADMISSION_DATE_STR}  |  Discharge: {DISCHARGE_DATE_STR}  |  "
        f"Ref ID: APL-MED-2026-{CLAIM_ID_LABEL}",
    )

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Patient Information", HEADER_STYLE))
    story.append(_patient_info_table())
    story.append(Spacer(1, 4 * mm))

    # Diagnosis block
    story.append(Paragraph("Diagnosis & Clinical Findings", HEADER_STYLE))
    diag_data = [
        ["Primary Diagnosis:", f"{DIAGNOSIS} (ICD-10: K35.80)"],
        ["Secondary Diagnosis:", "Nil"],
        ["Clinical Presentation:",
         "Patient presented to the emergency department with a 24-hour history of right "
         "iliac fossa pain, nausea, vomiting, and fever (38.6°C). Rebound tenderness and "
         "guarding noted on abdominal examination. Rovsing's sign positive."],
        ["Investigations:", "CBC: WBC 11.2 ×10³/μL (mild leukocytosis). CRP elevated. "
         "USG Abdomen: Dilated, non-compressible appendix (8 mm). CT Abdomen: "
         "Confirmed uncomplicated acute appendicitis. No perforation."],
    ]
    dt = Table(diag_data, colWidths=[45 * mm, 135 * mm])
    dt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(dt)
    story.append(Spacer(1, 4 * mm))

    # Procedure
    story.append(Paragraph("Surgical Procedure", HEADER_STYLE))
    proc_data = [
        ["Procedure:", PROCEDURE],
        ["Surgeon:", f"{DOCTOR_NAME}  ({DOCTOR_REG})"],
        ["Anaesthesiologist:", "Dr. Karthik Raj, MD (Anaesthesiology) | Reg. TN-MCI-18431"],
        ["Date of Surgery:", ADMISSION_DATE_STR],
        ["Duration:", "45 minutes"],
        ["Anaesthesia:", "General Anaesthesia (GA)"],
        ["Surgical Findings:",
         "Grossly inflamed appendix with hyperaemia, no perforation, no peritonitis. "
         "Appendix successfully excised via laparoscopic approach. Histopathology "
         "sample sent to lab (results consistent with acute appendicitis)."],
        ["Post-op Course:",
         "Uneventful. Patient tolerated oral feeds from POD1. IV antibiotics continued "
         "for 48 hours. Pain managed with analgesics. Wound clean and dry."],
        ["Discharge Condition:", "Stable. Patient ambulant. Vitals within normal limits."],
    ]
    pt = Table(proc_data, colWidths=[45 * mm, 135 * mm])
    pt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(pt)
    story.append(Spacer(1, 4 * mm))

    # Discharge instructions
    story.append(Paragraph("Discharge Instructions & Follow-up", HEADER_STYLE))
    story.append(Paragraph(
        "1. Continue prescribed antibiotics (Augmentin 625 mg BD × 5 days) and analgesics (Paracetamol 500 mg SOS).\n"
        "2. Keep wound dry. Dressing change every 48 hours at the nearest clinic.\n"
        "3. Avoid strenuous activity for 4 weeks. Light walking encouraged.\n"
        "4. Follow-up appointment: 1 week post-discharge for suture removal.\n"
        "5. Return to emergency if fever >38°C, wound discharge, or worsening pain.",
        BODY_STYLE,
    ))
    story.append(Spacer(1, 6 * mm))

    # Physician signature block
    story.append(_sep())
    story.append(Paragraph("Physician Declaration & Signature", HEADER_STYLE))
    sig_data = [
        ["Attending Surgeon:", f"{DOCTOR_NAME}"],
        ["Qualification:", "MBBS, MS (General Surgery)"],
        ["Medical Council Registration:", "TN-MCI-20842"],
        ["Hospital:", HOSPITAL_NAME],
        ["Date:", REPORT_DATE_STR],
        ["Digital / Physical Signature:", "[SIGNED & STAMPED]"],
    ]
    st = Table(sig_data, colWidths=[55 * mm, 125 * mm])
    st.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(st)

    doc.build(story)
    print(f"  ✅ Medical report generated →  {output_path}")


# ===========================================================================
# SCENARIO: FAILURE
# Defects injected:
#   Bill          — dated 45 days in the FUTURE (invalid) + no billing address
#   Blood report  — lab accreditation removed, patient name slightly different
#   Medical report — physician signature block omitted, date 6 weeks ago (>1 month)
# ===========================================================================

def generate_bill_failure(output_path: str):
    future_date = (TODAY + timedelta(days=45)).strftime("%d-%b-%Y")
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
                             topMargin=15*mm, bottomMargin=15*mm)
    story = _header_block(
        "HOSPITAL BILL / TAX INVOICE",
        f"Bill No: APL-FAIL-001  |  Bill Date: {future_date}  |  "
        f"Admission: {ADMISSION_DATE_STR}  |  Discharge: {DISCHARGE_DATE_STR}",
    )
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph("Patient Information", HEADER_STYLE))
    # Intentionally omit billing address row
    data = [
        ["Patient Name", PATIENT_NAME, "Policy No.", POLICY_NUMBER],
        ["Date of Birth", PATIENT_DOB, "Claim ID", CLAIM_ID_LABEL],
        # No address row — checklist: billing_address_present → FAIL
    ]
    t = Table(data, colWidths=[38*mm, 62*mm, 28*mm, 52*mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Charge Breakdown", HEADER_STYLE))
    charges = [
        ["S.No", "Description", "Amount (₹)"],
        ["1", "Room & Nursing (2 nights)", "10,000.00"],
        ["2", "Surgical Fee", "60,000.00"],
        ["", "TOTAL AMOUNT DUE (₹)", "70,000.00"],
    ]
    ct = Table(charges, colWidths=[15*mm, 120*mm, 45*mm])
    ct.setStyle(_table_style())
    story.append(ct)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        f"⚠ NOTE: This bill is intentionally dated in the future ({future_date}) "
        "and has no billing address — for FAILURE scenario testing.",
        SMALL,
    ))
    doc.build(story)
    print(f"  ❌ Failure bill generated      →  {output_path}")


def generate_blood_report_failure(output_path: str):
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
                             topMargin=15*mm, bottomMargin=15*mm)
    story = []
    # Lab header WITHOUT accreditation number — checklist: lab_name_present → FAIL
    story.append(Paragraph("City Diagnostics Centre", TITLE_STYLE))
    story.append(Paragraph("Branch: T. Nagar, Chennai – 600 017", SMALL))
    story.append(Paragraph("(Accreditation details not available)", SMALL))   # ← DEFECT
    story.append(Spacer(1, 3*mm))
    story.append(_sep())
    story.append(Paragraph("BLOOD TEST REPORT", HEADER_STYLE))
    story.append(Paragraph(
        f"Report Date: {REPORT_DATE_STR}  |  Ref ID: CDX-FAIL-001",
        SMALL,
    ))
    story.append(_sep())
    story.append(Spacer(1, 2*mm))
    # Name mismatch — checklist: full_name_present cross-doc → FAIL
    mismatch_data = [
        ["Patient Name", "R. Sharma", "Policy No.", POLICY_NUMBER],   # abbreviated name
        ["Date of Birth", PATIENT_DOB, "Claim ID", CLAIM_ID_LABEL],
    ]
    mt = Table(mismatch_data, colWidths=[38*mm, 62*mm, 28*mm, 52*mm])
    mt.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]))
    story.append(mt)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Complete Blood Count", HEADER_STYLE))
    cbc = [
        ["Test", "Result", "Reference Range", "Status"],
        ["Haemoglobin", "7.2", "13.0–17.0 g/dL", "⚠ LOW"],
        ["WBC", "18.5", "4.0–11.0 ×10³/μL", "⚠ HIGH"],
        ["Platelets", "0.9", "1.5–4.0 ×10⁵/μL", "⚠ LOW"],
    ]
    bt = Table(cbc, colWidths=[70*mm, 30*mm, 50*mm, 30*mm])
    bt.setStyle(_table_style())
    story.append(bt)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "⚠ NOTE: Lab accreditation missing, patient name abbreviated, "
        "multiple abnormal values — for FAILURE scenario testing.",
        SMALL,
    ))
    doc.build(story)
    print(f"  ❌ Failure blood report generated →  {output_path}")


def generate_medical_report_failure(output_path: str):
    # Date 6 weeks ago → outside 1-month window — checklist: dated_within_1_month → FAIL
    old_date = (TODAY - timedelta(days=42)).strftime("%d-%b-%Y")
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
                             topMargin=15*mm, bottomMargin=15*mm)
    story = _header_block(
        "MEDICAL REPORT",
        f"Report Date: {old_date}  |  Admission: {ADMISSION_DATE_STR}  |  Discharge: {DISCHARGE_DATE_STR}",
    )
    story.append(Spacer(1, 3*mm))
    story.append(_patient_info_table())
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Diagnosis & Treatment", HEADER_STYLE))
    story.append(Paragraph(
        f"Diagnosis: {DIAGNOSIS} (ICD-10: K35.80)\n"
        f"Procedure: {PROCEDURE}\n"
        "Clinical notes on file. Patient discharged in stable condition.",
        BODY_STYLE,
    ))
    story.append(Spacer(1, 6*mm))
    story.append(_sep())
    story.append(Paragraph("Physician Declaration", HEADER_STYLE))
    # Intentionally NO signature — checklist: physician_signature → FAIL
    story.append(Paragraph(
        "Attending Physician: [NAME WITHHELD]\n"
        "Signature: ___________________________  (NOT SIGNED)\n"
        f"Date: {old_date}",
        BODY_STYLE,
    ))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        f"⚠ NOTE: Report dated {old_date} (>1 month ago) and physician signature absent "
        "— for FAILURE scenario testing.",
        SMALL,
    ))
    doc.build(story)
    print(f"  ❌ Failure medical report generated →  {output_path}")


# ===========================================================================
# SCENARIO: INCOMPLETE  (only blood report present — bill + medical report missing)
# ===========================================================================

def generate_incomplete_scenario(out_dir: str):
    generate_blood_report(os.path.join(out_dir, "Apollo_Diagnostics_Blood_Report_Rahul_Sharma.pdf"))
    print(f"  ⚠️  Only blood report uploaded — Bill and Medical report intentionally absent.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    claim_id = sys.argv[1] if len(sys.argv) > 1 else "CLAIM_ID_192113"
    scenario  = sys.argv[2].lower() if len(sys.argv) > 2 else "success"

    if not claim_id.startswith("CLAIM_ID_"):
        print(f"⚠️  Warning: '{claim_id}' does not match CLAIM_ID_XXXXXX format.")

    out_dir = os.path.join(DIR, claim_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n📁 Output folder : {out_dir}")
    print(f"🧪 Scenario      : {scenario}\n")

    if scenario == "success":
        generate_bill(os.path.join(out_dir, "Apollo_Hospital_Bill_CLM-2025-03-00912.pdf"))
        generate_blood_report(os.path.join(out_dir, "Apollo_Diagnostics_Blood_Report_Rahul_Sharma.pdf"))
        generate_medical_report(os.path.join(out_dir, "Apollo_Surgical_Report_Rahul_Sharma.pdf"))
        print(f"\n✅ SUCCESS scenario PDFs generated for {claim_id}")
        print("   Expected result: Ready | Low risk | ~85-100 score\n")

    elif scenario == "failure":
        generate_bill_failure(os.path.join(out_dir, "Apollo_Hospital_Bill_CLM-2025-03-00912.pdf"))
        generate_blood_report_failure(os.path.join(out_dir, "Apollo_Diagnostics_Blood_Report_Rahul_Sharma.pdf"))
        generate_medical_report_failure(os.path.join(out_dir, "Apollo_Surgical_Report_Rahul_Sharma.pdf"))
        print(f"\n❌ FAILURE scenario PDFs generated for {claim_id}")
        print("   Expected result: Not Ready | Critical/High risk | low score\n")

    elif scenario == "incomplete":
        generate_incomplete_scenario(out_dir)
        print(f"\n⚠️  INCOMPLETE scenario PDF generated for {claim_id}")
        print("   Expected result: Not Ready | Critical risk (missing Bill + Medical report)\n")

    else:
        print(f"❌ Unknown scenario '{scenario}'. Choose: success | failure | incomplete")
        sys.exit(1)


if __name__ == "__main__":
    main()
