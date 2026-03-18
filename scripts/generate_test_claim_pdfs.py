from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

styles = getSampleStyleSheet()

TITLE = ParagraphStyle("title", parent=styles["Heading1"], fontSize=16, textColor=colors.red)
HEADER = ParagraphStyle("header", parent=styles["Heading2"], fontSize=12)
BODY = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14)

def generate_rejection_report(output_path):
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=15*mm, bottomMargin=15*mm)

    story = []

    # Title
    story.append(Paragraph("CLAIM REJECTION REPORT", TITLE))
    story.append(Spacer(1, 6))

    # Status
    story.append(Paragraph("<b>Status: REJECTED</b>", BODY))
    story.append(Spacer(1, 10))

    # Patient Details
    story.append(Paragraph("Policy & Patient Details", HEADER))
    patient_data = [
        ["Patient Name", "Ananya Nair"],
        ["Age / Gender", "29 / Female"],
        ["Policy Number", "HLT-77451209"],
        ["Claim ID", "CLM-2026-22451"]
    ]
    table = Table(patient_data, colWidths=[120, 300])
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold")
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    # Hospital Details
    story.append(Paragraph("Hospital Details", HEADER))
    hospital_data = [
        ["Hospital", "Elite Aesthetic Clinic"],
        ["Admission Date", "05 March 2026"],
        ["Discharge Date", "06 March 2026"]
    ]
    table = Table(hospital_data, colWidths=[120, 300])
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold")
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    # Treatment Details
    story.append(Paragraph("Treatment Details", HEADER))
    story.append(Paragraph(
        "<b>Procedure:</b> Rhinoplasty (Cosmetic Nose Reshaping)<br/>"
        "<b>Claim Amount:</b> ₹2,40,000<br/>"
        "<b>Medical Necessity:</b> Not established",
        BODY
    ))
    story.append(Spacer(1, 10))

    # Assessment
    story.append(Paragraph("Claim Assessment", HEADER))
    story.append(Paragraph(
        "- Procedure identified as cosmetic<br/>"
        "- No medical necessity justification<br/>"
        "- Doctor notes confirm elective intent",
        BODY
    ))
    story.append(Spacer(1, 10))

    # Policy Clause
    story.append(Paragraph("Policy Clause Applied", HEADER))
    story.append(Paragraph(
        "Cosmetic treatments are excluded unless medically required due to illness, injury, or congenital condition.",
        BODY
    ))
    story.append(Spacer(1, 10))

    # Final Decision
    story.append(Paragraph("Final Decision", HEADER))
    story.append(Paragraph(
        "<b>Claim Status:</b> Rejected<br/>"
        "<b>Payable Amount:</b> ₹0<br/>"
        "<b>Reason:</b> Cosmetic procedure not covered",
        BODY
    ))
    story.append(Spacer(1, 20))

    # Footer
    story.append(Paragraph("Authorized by: Claims Team, Pure Insurance", BODY))

    doc.build(story)

    print(f"❌ Rejection report generated → {output_path}")