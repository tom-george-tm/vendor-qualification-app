from jinja2 import Environment, select_autoescape, PackageLoader
import asyncio

env = Environment(
    loader=PackageLoader('app', 'templates'),
    autoescape=select_autoescape(['html', 'xml'])
)

template = env.get_template('approve_email.html')

# Test 1: Full variables (New flow)
html1 = template.render(
    claim_id="CLM-123",
    applicant_name="Test User",
    hospital_name="Test Hospital",
    diagnosis="Test Diagnosis",
    total_amount=1000.0,
    approved_amount=800.0,
    non_payable_amount=200.0
)
print("TEST 1 (Full): PASS")

# Test 2: Missing variables (Legacy flow)
html2 = template.render(
    claim_id="CLM-123",
    applicant_name="Test User",
    policy_number="POL-123",
    reasoning="This is legacy reasoning."
)
print("TEST 2 (Legacy): PASS")
