import base64
import time
import httpx
from typing import List, Dict, Any, Optional
from .config import settings
from jinja2 import Environment, select_autoescape, PackageLoader


# Jinja2 environment for templates
env = Environment(
    loader=PackageLoader('app', 'templates'),
    autoescape=select_autoescape(['html', 'xml'])
)


class GmailAPI:
    """
    Gmail API logic for sending emails and refreshing tokens.
    """
    GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
    TOKEN_URL = "https://oauth2.googleapis.com/token"

    def __init__(self):
        self.client_id = settings.GMAIL_CLIENT_ID
        self.client_secret = settings.GMAIL_CLIENT_SECRET
        self.refresh_token = settings.GMAIL_REFRESH_TOKEN
        self.access_token = settings.GMAIL_ACCESS_TOKEN
        self.expire_time = settings.GMAIL_EXPIRE_TIME

    async def _refresh_access_token(self) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.TOKEN_URL,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()

        data = response.json()
        self.access_token = data["access_token"]
        self.expire_time = int(time.time()) + int(data.get("expires_in", 3600))
        
        # Note: We don't persist expire_time/access_token back to .env here, 
        # but the Gmail Tool usually handles this or does it per execution.
        return self.access_token

    async def _get_valid_access_token(self) -> str:
        if not self.access_token or time.time() >= self.expire_time - 60:
            return await self._refresh_access_token()
        return self.access_token

    async def send_email(self, recipients: List[str], subject: str, html_body: str):
        token = await self._get_valid_access_token()
        
        # Build the message
        # Basic MIME-like format for Gmail 'raw' sending
        to_header = ", ".join(recipients)
        message_parts = [
            f"To: {to_header}",
            f"Subject: {subject}",
            "MIME-Version: 1.0",
            "Content-Type: text/html; charset=utf-8",
            "",
            html_body
        ]
        raw_message = "\n".join(message_parts)

        encoded_message = base64.urlsafe_b64encode(
            raw_message.encode("utf-8")
        ).decode("utf-8")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.GMAIL_SEND_URL,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={"raw": encoded_message},
            )
            
        if response.status_code != 200:
            print(f"Gmail Error: {response.text}")
        response.raise_for_status()
        return response.json()


class Email:
    def __init__(self, name: str, url: str, email: Optional[List[str]] = None):
        self.name = name
        self.url = url
        self.email = email or [settings.GMAIL_TO]
        self.gmail_api = GmailAPI()

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
            await self.gmail_api.send_email(
                recipients=self.email,
                subject=subject,
                html_body=html
            )
        except Exception as e:
            print(f"Failed to send email via Gmail API: {e}")

    async def send_workflow_notification(self, filename: str, decision: str, reasoning: str):
        await self.sendMail(
            subject=f'Workflow Update: {decision} for {filename}',
            template_name='workflow_notification',
            filename=filename,
            decision=decision,
            reasoning=reasoning
        )
