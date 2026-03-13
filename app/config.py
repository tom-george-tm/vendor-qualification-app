from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import EmailStr
from typing import Optional


class Settings(BaseSettings):
    DATABASE_URL: str
    MONGO_INITDB_DATABASE: str
    CLIENT_ORIGIN: str

    # SMTP (Deprecated but kept for now)
    EMAIL_HOST: Optional[str] = None
    EMAIL_PORT: Optional[int] = None
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    EMAIL_FROM: Optional[EmailStr] = None

    # Gmail API Settings (all optional — email notifications skipped if not set)
    GMAIL_CLIENT_ID: Optional[str] = None
    GMAIL_CLIENT_SECRET: Optional[str] = None
    GMAIL_ACCESS_TOKEN: Optional[str] = None
    GMAIL_REFRESH_TOKEN: Optional[str] = None
    GMAIL_EXPIRE_TIME: int = 0
    GMAIL_TO: Optional[EmailStr] = None

    # OpenAI settings
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Legacy workflow settings (kept for reference)
    WORKFLOW_API_URL: str = "http://localhost:8000/workflow/163f55aa-4a92-4164-94cb-a4210e1d7509"
    NOTIFICATION_EMAIL: EmailStr = "admin@example.com"
    
    # OpenAI Settings
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_file='./.env', extra='ignore')


settings = Settings()
