import smtplib
from email.message import EmailMessage
import asyncio
from typing import List, Optional
from .config import settings
from jinja2 import Environment, select_autoescape, PackageLoader


# Jinja2 environment for templates
env = Environment(
    loader=PackageLoader('app', 'templates'),
    autoescape=select_autoescape(['html', 'xml'])
)


class SMTPEmailAPI:
    """
    Standard SMTP logic for sending emails.
    """
    def __init__(self):
        self.host = "smtp.gmail.com"
        self.port = 587
        self.username = "pridhviraj9248@gmail.com"
        self.password = "njoi wrbt kkoc puls"
        self.from_email = self.username

    def _send_email_sync(self, recipients: List[str], subject: str, html_body: str):
        if not self.username or not self.password:
            print("SMTP Error: EMAIL_USERNAME or EMAIL_PASSWORD is not set in environment.")
            return

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = self.from_email
        msg['To'] = ", ".join(recipients)
        msg.set_content("Please enable HTML to view this email.")
        msg.add_alternative(html_body, subtype='html')

        try:
            with smtplib.SMTP(self.host, self.port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                print(f"Email successfully sent to {msg['To']}")
        except Exception as e:
            print(f"SMTP Error: Failed to send email: {e}")
            raise e

    async def send_email(self, recipients: List[str], subject: str, html_body: str):
        # Run the synchronous SMTP call in a separate thread so it doesn't block FastAPI
        await asyncio.to_thread(self._send_email_sync, recipients, subject, html_body)


class Email:
    def __init__(self, name: str, url: str, email: Optional[List[str]] = None):
        self.name = name
        self.url = url
        self.email = email or [settings.GMAIL_TO]
        self.smtp_api = SMTPEmailAPI()

    async def sendMail(self, subject: str, template_name: str, **kwargs):
        # Generate the HTML template
        template = env.get_template(f'{template_name}.html')

        html = template.render(
            url=self.url,
            first_name=self.name,
            subject=subject,
            **kwargs
        )

        try:
            await self.smtp_api.send_email(
                recipients=self.email,
                subject=subject,
                html_body=html
            )
        except Exception as e:
            print(f"Failed to send email via SMTP API: {e}")

    async def send_approval_email(self, claim_id: str, applicant_name: str, policy_number: str, reasoning: str = ""):
        await self.sendMail(
            subject=f'Claim Approved: {claim_id}',
            template_name='approve_email',
            claim_id=claim_id,
            applicant_name=applicant_name,
            policy_number=policy_number,
            reasoning=reasoning
        )

    async def send_rejection_email(
        self,
        claim_id: str,
        applicant_name: str,
        policy_number: str,
        reasoning: str,
        rejection_reason: str = "",
        policy_clause: str = "",
        clause_details: str = "",
        diagnosis: str = "",
    ):
        await self.sendMail(
            subject=f'Claim Decision Notice – Claim {claim_id}',
            template_name='reject_email',
            claim_id=claim_id,
            applicant_name=applicant_name,
            policy_number=policy_number,
            reasoning=reasoning,
            rejection_reason=rejection_reason or reasoning,
            policy_clause=policy_clause,
            clause_details=clause_details,
            diagnosis=diagnosis,
        )

    async def send_provider_intimation_email(self, claim_id: str, applicant_name: str, total_amount: float, approved_amount: float, provider_name: str = "Healthcare Provider"):
        """
        Sends an email notifying the healthcare provider that the claim has been approved via straight-through processing.
        """
        await self.sendMail(
            subject=f'Claim Approval Notification – Claim {claim_id}',
            template_name='provider_intimation',
            claim_id=claim_id,
            applicant_name=applicant_name,
            total_amount=total_amount,
            approved_amount=approved_amount,
            provider_name=provider_name
        )

    async def send_missing_info_email(self, claim_id: str, missing_documents: List[str], provider_name: str = "Healthcare Provider"):
        """
        Sends an email requesting missing documents from the healthcare provider.
        This handles the scenario when the AI detects missing data.
        """
        await self.sendMail(
            subject=f'Action Required: Missing Documents for Claim {claim_id}',
            template_name='missing_documents',
            claim_id=claim_id,
            missing_documents=missing_documents,
            provider_name=provider_name
        )

    async def send_provider_rejection_email(
        self,
        claim_id: str,
        applicant_name: str,
        rejection_reason: str,
        policy_clause: str = "",
        clause_details: str = "",
        diagnosis: str = "",
        claimed_amount: Optional[float] = None,
        provider_name: str = "Healthcare Provider",
    ):
        """
        Sends a rejection notification to the healthcare provider.
        """
        await self.sendMail(
            subject=f'Claim Rejection Notification – Claim {claim_id}',
            template_name='provider_rejection',
            claim_id=claim_id,
            applicant_name=applicant_name,
            rejection_reason=rejection_reason,
            policy_clause=policy_clause,
            clause_details=clause_details,
            diagnosis=diagnosis,
            claimed_amount=claimed_amount,
            provider_name=provider_name,
        )

